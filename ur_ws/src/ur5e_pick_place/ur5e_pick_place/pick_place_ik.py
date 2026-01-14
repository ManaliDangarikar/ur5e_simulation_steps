#!/usr/bin/env python3
"""
Compute IK with MoveIt (/compute_ik) for a target (x,y,z) and, if successful,
execute the resulting joint positions via FollowJointTrajectory.

Usage:
  ./ur5e_ik_and_execute.py
  ./ur5e_ik_and_execute.py --x 0.50 --y 0.00 --z 0.40
  ./ur5e_ik_and_execute.py --controller scaled   # or joint
"""

import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from builtin_interfaces.msg import Duration
from action_msgs.msg import GoalStatus
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance

from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped


def _duration(sec: float) -> Duration:
    d = Duration()
    d.sec = int(sec)
    d.nanosec = int((sec - int(sec)) * 1e9)
    return d


class UR5eIKAndExecute(Node):
    def __init__(
        self,
        group_name: str,
        ik_link_name: str,
        frame_id: str,
        ik_service: str,
        action_name: str,
        motion_time: float,
    ):
        super().__init__("ur5e_ik_and_execute")

        self.group_name = group_name
        self.ik_link_name = ik_link_name
        self.frame_id = frame_id
        self.ik_service = ik_service
        self.motion_time = motion_time

        # IK service client
        self.ik_client = self.create_client(GetPositionIK, self.ik_service)
        self.get_logger().info(f"Waiting for IK service: {self.ik_service}")
        if not self.ik_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f"IK service not available: {self.ik_service}")

        # Trajectory action client
        self.action_name = action_name
        self.action_client = ActionClient(self, FollowJointTrajectory, self.action_name)
        self.get_logger().info(f"Waiting for trajectory action: {self.action_name}")
        if not self.action_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(
                f"Trajectory action not available: {self.action_name}\n"
                f"Tip: check `ros2 action list | grep follow_joint_trajectory`"
            )

    def compute_ik(self, x: float, y: float, z: float) -> GetPositionIK.Response:
        ps = PoseStamped()
        ps.header.frame_id = self.frame_id
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = float(z)

        # Default orientation (identity). Change if you need tool pointing down.
        ps.pose.orientation.x = 0.0
        ps.pose.orientation.y = 0.0
        ps.pose.orientation.z = 0.0
        ps.pose.orientation.w = 1.0

        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.ik_link_name
        req.ik_request.pose_stamped = ps
        req.ik_request.timeout = _duration(1.0)
        req.ik_request.avoid_collisions = False

        fut = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        if fut.result() is None:
            raise RuntimeError("IK service call failed (no response).")
        return fut.result()

    def execute_joint_positions(self, joint_names, joint_positions):
        # Build trajectory
        traj = JointTrajectory()
        traj.joint_names = list(joint_names)

        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in joint_positions]
        pt.time_from_start = _duration(self.motion_time)
        traj.points = [pt]

        # Build action goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        # Optional tolerances (you can relax/tighten as needed)
        # Using modest defaults; controllers may ignore if not supported.
        tol = JointTolerance()
        tol.position = 0.01  # rad
        tol.velocity = 0.1   # rad/s
        tol.acceleration = 0.0
        goal.goal_tolerance = [tol]
        goal.path_tolerance = [tol]
        goal.goal_time_tolerance = _duration(1.0)

        self.get_logger().info("Sending FollowJointTrajectory goal...")
        send_fut = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut, timeout_sec=5.0)

        goal_handle = send_fut.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Trajectory goal was rejected by the controller.")

        self.get_logger().info("Goal accepted. Waiting for result...")
        result_fut = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_fut)

        wrapped = result_fut.result()
        status = wrapped.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("✅ Motion execution succeeded.")
        else:
            self.get_logger().error(f"❌ Motion failed. GoalStatus={status}")
        return status

    def run_once(self, x: float, y: float, z: float) -> int:
        res = self.compute_ik(x, y, z)
        code = res.error_code.val

        if code != 1:
            self.get_logger().error(f"IK failed. MoveItErrorCodes.val={code}")
            return code

        js = res.solution.joint_state
        self.get_logger().info(
            "IK success. Executing joints:\n"
            + "\n".join(f"  {n}: {p:.6f}" for n, p in zip(js.name, js.position))
        )

        # Small delay can help if you just launched nodes
        time.sleep(0.2)

        self.execute_joint_positions(js.name, js.position)
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, default=0.5, help="Target X in base frame (default: 0.5)")
    parser.add_argument("--y", type=float, default=0.0, help="Target Y in base frame (default: 0.0)")
    parser.add_argument("--z", type=float, default=0.4, help="Target Z in base frame (default: 0.4)")

    parser.add_argument("--group", type=str, default="ur_manipulator", help="MoveIt planning group")
    parser.add_argument("--ik-link", type=str, default="tool0", help="IK link name (end-effector link)")
    parser.add_argument("--frame", type=str, default="base_link", help="Pose frame_id")
    parser.add_argument("--ik-service", type=str, default="/compute_ik", help="IK service name")

    parser.add_argument(
        "--controller",
        choices=["scaled", "joint"],
        default="scaled",
        help="Which controller action to use",
    )
    parser.add_argument("--time", type=float, default=3.0, help="Motion time in seconds (default: 3.0)")

    args = parser.parse_args()

    action_name = (
        "/scaled_joint_trajectory_controller/follow_joint_trajectory"
        if args.controller == "scaled"
        else "/joint_trajectory_controller/follow_joint_trajectory"
    )

    rclpy.init()
    try:
        node = UR5eIKAndExecute(
            group_name=args.group,
            ik_link_name=args.ik_link,
            frame_id=args.frame,
            ik_service=args.ik_service,
            action_name=action_name,
            motion_time=args.time,
        )
        node.run_once(args.x, args.y, args.z)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()

