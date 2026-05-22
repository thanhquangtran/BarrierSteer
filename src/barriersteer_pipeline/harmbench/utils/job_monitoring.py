"""
Slurm job monitoring utilities for HarmBench pipeline.

This module provides utilities for monitoring and managing Slurm jobs,
including waiting for job completion and checking job status.
"""

import logging
import subprocess
import time
from typing import List


class SlurmJobMonitor:
    """Utility class for monitoring Slurm jobs"""

    def __init__(self, config: dict):
        """
        Initialize job monitor

        Args:
            config: Pipeline configuration containing Slurm settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Get configuration parameters
        slurm_config = config.get("slurm", {})
        self.check_interval = slurm_config.get("check_interval_seconds", 60)
        self.timeout_minutes = slurm_config.get("job_timeout_minutes", 240)

    def wait_for_jobs(self, job_ids: List[str], job_type: str = "jobs") -> bool:
        """
        Monitor Slurm jobs until completion

        Args:
            job_ids: List of Slurm job IDs to monitor
            job_type: Description of job type for logging

        Returns:
            True if all jobs completed successfully, False if any failed
        """
        if not job_ids:
            return True

        self.logger.info(f"Monitoring {len(job_ids)} {job_type}: {job_ids}")

        max_checks = (self.timeout_minutes * 60) // self.check_interval

        for check_count in range(max_checks):
            try:
                # Check job status using squeue for running jobs
                running_jobs = self._get_running_jobs(job_ids)
                completed_jobs = []
                failed_jobs = []

                # For jobs not in squeue, check sacct for completion status
                for job_id in job_ids:
                    if job_id not in running_jobs:
                        status = self._get_job_completion_status(job_id)
                        if status == "COMPLETED":
                            completed_jobs.append(job_id)
                        elif status in [
                            "FAILED",
                            "CANCELLED",
                            "TIMEOUT",
                            "NODE_FAIL",
                        ]:
                            failed_jobs.append(job_id)

                # Log progress
                total_jobs = len(job_ids)
                completed_count = len(completed_jobs)
                running_count = len(running_jobs)
                failed_count = len(failed_jobs)

                self.logger.info(
                    f"{job_type} status: {completed_count}/{total_jobs} completed, "
                    f"{running_count} running, {failed_count} failed"
                )

                # Check if any jobs failed
                if failed_jobs:
                    self.logger.error(f"Failed {job_type}: {failed_jobs}")
                    return False

                # Check if all jobs completed
                if completed_count == total_jobs:
                    self.logger.info(f"All {job_type} completed successfully")
                    return True

                # Wait before next check
                time.sleep(self.check_interval)

            except Exception as e:
                self.logger.warning(f"Error checking job status: {e}")
                time.sleep(self.check_interval)

        # Timeout reached
        remaining_jobs = [jid for jid in job_ids if jid not in completed_jobs]
        self.logger.error(
            f"Timeout waiting for {job_type}. Remaining jobs: {remaining_jobs}"
        )
        return False

    def _get_running_jobs(self, job_ids: List[str]) -> List[str]:
        """Get list of job IDs that are currently running"""
        if not job_ids:
            return []

        try:
            # Use squeue to check running jobs
            cmd = [
                "squeue",
                "-j",
                ",".join(job_ids),
                "--noheader",
                "--format=%i",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                running_jobs = [
                    line.strip()
                    for line in result.stdout.strip().split("\n")
                    if line.strip()
                ]
                return running_jobs
            else:
                # If squeue fails, assume no jobs are running
                return []

        except Exception as e:
            self.logger.warning(f"Error checking running jobs: {e}")
            return []

    def _get_job_completion_status(self, job_id: str) -> str:
        """Get completion status of a job using sacct"""
        try:
            # Use sacct to check job completion status
            cmd = ["sacct", "-j", job_id, "--noheader", "--format=State", "-P"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0 and result.stdout.strip():
                # Get the first status (main job, not job steps)
                status = result.stdout.strip().split("\n")[0].strip()
                return status
            else:
                # If sacct fails or no output, assume job is still running
                return "RUNNING"

        except Exception as e:
            self.logger.warning(
                f"Error checking job completion status for {job_id}: {e}"
            )
            return "UNKNOWN"
