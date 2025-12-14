"""
Performance monitoring utilities for tracking execution time of different stages.
"""
import time
import logging
from typing import Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and log performance metrics for different stages."""
    
    def __init__(self, name: str = "PerformanceMonitor"):
        self.name = name
        self.stages: Dict[str, float] = {}
        self.start_time: Optional[float] = None
        self.current_stage: Optional[str] = None
        self.stage_start_time: Optional[float] = None
    
    def start(self):
        """Start overall timing."""
        self.start_time = time.time()
        self.stages.clear()
        logger.info(f"[{self.name}] ⏱️  开始性能监控")
    
    def start_stage(self, stage_name: str):
        """Start timing a specific stage."""
        if self.stage_start_time is not None and self.current_stage:
            # End previous stage if exists
            elapsed = time.time() - self.stage_start_time
            self.stages[self.current_stage] = elapsed
            logger.info(f"[{self.name}] ⏱️  阶段 '{self.current_stage}' 耗时: {elapsed:.3f}秒")
        
        self.current_stage = stage_name
        self.stage_start_time = time.time()
        logger.info(f"[{self.name}] ⏱️  开始阶段: {stage_name}")
    
    def end_stage(self, stage_name: Optional[str] = None):
        """End timing a specific stage."""
        if self.stage_start_time is None:
            return
        
        stage = stage_name or self.current_stage
        if stage:
            elapsed = time.time() - self.stage_start_time
            if stage not in self.stages:
                self.stages[stage] = 0
            self.stages[stage] += elapsed
            logger.info(f"[{self.name}] ⏱️  阶段 '{stage}' 耗时: {elapsed:.3f}秒")
        
        self.current_stage = None
        self.stage_start_time = None
    
    def end(self):
        """End overall timing and log summary."""
        if self.stage_start_time is not None and self.current_stage:
            self.end_stage()
        
        if self.start_time is None:
            return
        
        total_time = time.time() - self.start_time
        
        logger.info(f"[{self.name}] " + "=" * 60)
        logger.info(f"[{self.name}] 📊 性能监控总结:")
        logger.info(f"[{self.name}]   总耗时: {total_time:.3f}秒")
        
        if self.stages:
            logger.info(f"[{self.name}]   各阶段耗时:")
            sorted_stages = sorted(self.stages.items(), key=lambda x: x[1], reverse=True)
            for stage, elapsed in sorted_stages:
                percentage = (elapsed / total_time * 100) if total_time > 0 else 0
                logger.info(f"[{self.name}]     - {stage}: {elapsed:.3f}秒 ({percentage:.1f}%)")
            
            # Highlight the slowest stage
            if sorted_stages:
                slowest_stage, slowest_time = sorted_stages[0]
                logger.info(f"[{self.name}]   ⚠️  最慢阶段: {slowest_stage} ({slowest_time:.3f}秒, {slowest_time/total_time*100:.1f}%)")
        
        logger.info(f"[{self.name}] " + "=" * 60)
        
        self.start_time = None
        self.stages.clear()
    
    @contextmanager
    def stage(self, stage_name: str):
        """Context manager for timing a stage."""
        self.start_stage(stage_name)
        try:
            yield
        finally:
            self.end_stage(stage_name)
    
    def get_total_time(self) -> float:
        """Get total elapsed time."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def get_stage_time(self, stage_name: str) -> float:
        """Get elapsed time for a specific stage."""
        return self.stages.get(stage_name, 0.0)
