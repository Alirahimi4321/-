"""
Project AWARENESS - Memory Controller Agent
Manages the three-tier memory hierarchy with adaptive loading strategies.
"""

import asyncio
import time
import mmap
import pickle
import zlib
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import OrderedDict
import hashlib
import os
from pathlib import Path
from enum import Enum

from agents.base_agent import BaseAgent, AgentMessage
from core.config import AwarenessConfig


class MemoryTier(Enum):
    """Memory tier levels."""
    HOT = "hot"      # RAM-based LRU cache
    WARM = "warm"    # ZRAM compression
    COLD = "cold"    # Flash storage


class LoadingStrategy(Enum):
    """Memory loading strategies."""
    FULL = "full"        # Load entire object
    MMAP = "mmap"        # Memory-mapped file
    LAYERED = "layered"  # Progressive loading


@dataclass
class MemoryItem:
    """Represents an item in memory."""
    key: str
    data: Any
    size: int
    tier: MemoryTier
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    compressed: bool = False
    checksum: str = ""
    

@dataclass
class MemoryStats:
    """Memory usage statistics."""
    hot_size: int = 0
    warm_size: int = 0
    cold_size: int = 0
    total_items: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    

class MemoryControllerAgent(BaseAgent):
    """
    Memory Controller Agent implementing three-tier memory hierarchy.
    Manages Hot (RAM-LRU), Warm (ZRAM), and Cold (Flash) storage tiers.
    """
    
    def __init__(self, name: str, config: AwarenessConfig, agent_config: Dict[str, Any]):
        super().__init__(name, config, agent_config)
        
        # Memory tiers
        self.hot_cache: OrderedDict[str, MemoryItem] = OrderedDict()
        self.warm_cache: Dict[str, MemoryItem] = {}
        self.cold_storage: Dict[str, str] = {}  # key -> file path
        
        # Configuration
        self.hot_cache_size = self.config.memory.hot_cache_size_mb * 1024 * 1024
        self.warm_cache_size = self.config.memory.warm_cache_size_mb * 1024 * 1024
        self.cold_storage_path = Path(self.config.data_dir) / "cold_storage"
        
        # Current usage
        self.hot_usage = 0
        self.warm_usage = 0
        self.cold_usage = 0
        
        # Statistics
        self.stats = MemoryStats()
        
        # Memory management
        self.memory_lock = asyncio.Lock()
        self.compression_level = 6  # zlib compression level
        
        # Background tasks
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        
        # Memory mapping
        self.mmap_files: Dict[str, mmap.mmap] = {}
        
        # Register handlers
        self._register_memory_handlers()
        
    async def _initialize(self):
        """Initialize the memory controller."""
        self.logger.info("MemoryControllerAgent initializing...")
        
        # Create cold storage directory
        self.cold_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing cold storage index
        await self._load_cold_storage_index()
        
        # Initialize memory tiers
        await self._initialize_memory_tiers()
        
        self.logger.info("MemoryControllerAgent initialized successfully")
        
    async def _shutdown(self):
        """Shutdown the memory controller."""
        self.logger.info("MemoryControllerAgent shutting down...")
        
        # Save cold storage index
        await self._save_cold_storage_index()
        
        # Clean up memory mappings
        await self._cleanup_memory_mappings()
        
        # Final cleanup
        await self._cleanup_memory_tiers()
        
        self.logger.info("MemoryControllerAgent shutdown complete")
        
    def _register_memory_handlers(self):
        """Register memory-specific message handlers."""
        self.register_handler("memory_store", self._handle_memory_store)
        self.register_handler("memory_retrieve", self._handle_memory_retrieve)
        self.register_handler("memory_delete", self._handle_memory_delete)
        self.register_handler("memory_stats", self._handle_memory_stats)
        self.register_handler("memory_search", self._handle_memory_search)
        self.register_handler("memory_compact", self._handle_memory_compact)
        
    def _get_background_tasks(self) -> List:
        """Get memory-specific background tasks."""
        return [
            self._memory_cleanup_loop(),
            self._memory_optimization_loop(),
            self._memory_monitoring_loop(),
        ]
        
    async def _handle_memory_store(self, message: AgentMessage):
        """Handle memory store requests."""
        try:
            key = message.payload.get("key")
            data = message.payload.get("data")
            tier_hint = message.payload.get("tier", "auto")
            
            if not key or data is None:
                await self._send_error_response(message, "Missing key or data")
                return
                
            # Store in memory
            success = await self.store(key, data, tier_hint)
            
            await self._send_response(message, {
                "status": "success" if success else "failed",
                "key": key,
                "tier": self._get_item_tier(key)
            })
            
        except Exception as e:
            self.logger.error(f"Error handling memory store: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_memory_retrieve(self, message: AgentMessage):
        """Handle memory retrieve requests."""
        try:
            key = message.payload.get("key")
            loading_strategy = message.payload.get("loading_strategy", "full")
            
            if not key:
                await self._send_error_response(message, "Missing key")
                return
                
            # Retrieve from memory
            data = await self.retrieve(key, LoadingStrategy(loading_strategy))
            
            if data is not None:
                await self._send_response(message, {
                    "status": "success",
                    "key": key,
                    "data": data,
                    "tier": self._get_item_tier(key)
                })
            else:
                await self._send_response(message, {
                    "status": "not_found",
                    "key": key
                })
                
        except Exception as e:
            self.logger.error(f"Error handling memory retrieve: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_memory_delete(self, message: AgentMessage):
        """Handle memory delete requests."""
        try:
            key = message.payload.get("key")
            
            if not key:
                await self._send_error_response(message, "Missing key")
                return
                
            # Delete from memory
            success = await self.delete(key)
            
            await self._send_response(message, {
                "status": "success" if success else "not_found",
                "key": key
            })
            
        except Exception as e:
            self.logger.error(f"Error handling memory delete: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_memory_stats(self, message: AgentMessage):
        """Handle memory statistics requests."""
        try:
            stats = await self.get_stats()
            
            await self._send_response(message, {
                "status": "success",
                "stats": stats
            })
            
        except Exception as e:
            self.logger.error(f"Error handling memory stats: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_memory_search(self, message: AgentMessage):
        """Handle memory search requests."""
        try:
            query = message.payload.get("query")
            max_results = message.payload.get("max_results", 10)
            
            if not query:
                await self._send_error_response(message, "Missing query")
                return
                
            # Search memory
            results = await self.search(query, max_results)
            
            await self._send_response(message, {
                "status": "success",
                "results": results,
                "count": len(results)
            })
            
        except Exception as e:
            self.logger.error(f"Error handling memory search: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_memory_compact(self, message: AgentMessage):
        """Handle memory compaction requests."""
        try:
            # Perform memory compaction
            await self._compact_memory()
            
            await self._send_response(message, {
                "status": "success",
                "message": "Memory compaction completed"
            })
            
        except Exception as e:
            self.logger.error(f"Error handling memory compact: {e}")
            await self._send_error_response(message, str(e))
            
    async def store(self, key: str, data: Any, tier_hint: str = "auto") -> bool:
        """
        Store data in memory with automatic tier selection.
        
        Args:
            key: Unique identifier for the data
            data: Data to store
            tier_hint: Tier preference (hot, warm, cold, auto)
            
        Returns:
            True if stored successfully
        """
        async with self.memory_lock:
            try:
                # Create memory item
                serialized_data = pickle.dumps(data)
                item = MemoryItem(
                    key=key,
                    data=data,
                    size=len(serialized_data),
                    tier=self._select_optimal_tier(len(serialized_data), tier_hint),
                    checksum=hashlib.sha256(serialized_data).hexdigest()
                )
                
                # Remove existing item if present
                await self._remove_item(key)
                
                # Store in appropriate tier
                if item.tier == MemoryTier.HOT:
                    return await self._store_hot(key, item)
                elif item.tier == MemoryTier.WARM:
                    return await self._store_warm(key, item)
                else:
                    return await self._store_cold(key, item)
                    
            except Exception as e:
                self.logger.error(f"Error storing {key}: {e}")
                return False
                
    async def retrieve(self, key: str, strategy: LoadingStrategy = LoadingStrategy.FULL) -> Optional[Any]:
        """
        Retrieve data from memory with specified loading strategy.
        
        Args:
            key: Unique identifier for the data
            strategy: Loading strategy to use
            
        Returns:
            Retrieved data or None if not found
        """
        async with self.memory_lock:
            try:
                # Check hot cache first
                if key in self.hot_cache:
                    item = self.hot_cache[key]
                    item.accessed_at = time.time()
                    item.access_count += 1
                    
                    # Move to end (most recently used)
                    self.hot_cache.move_to_end(key)
                    
                    self.stats.cache_hits += 1
                    return item.data
                    
                # Check warm cache
                if key in self.warm_cache:
                    item = self.warm_cache[key]
                    data = await self._decompress_data(item.data)
                    
                    # Promote to hot cache if frequently accessed
                    if item.access_count > 5:
                        await self._promote_to_hot(key, data)
                    
                    item.accessed_at = time.time()
                    item.access_count += 1
                    
                    self.stats.cache_hits += 1
                    return data
                    
                # Check cold storage
                if key in self.cold_storage:
                    data = await self._load_cold_data(key, strategy)
                    
                    if data is not None:
                        # Promote to warm cache
                        await self._promote_to_warm(key, data)
                        
                        self.stats.cache_hits += 1
                        return data
                        
                # Not found
                self.stats.cache_misses += 1
                return None
                
            except Exception as e:
                self.logger.error(f"Error retrieving {key}: {e}")
                self.stats.cache_misses += 1
                return None
                
    async def delete(self, key: str) -> bool:
        """
        Delete data from memory.
        
        Args:
            key: Unique identifier for the data
            
        Returns:
            True if deleted successfully
        """
        async with self.memory_lock:
            return await self._remove_item(key)
            
    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for items in memory.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of matching items
        """
        results = []
        
        # Search hot cache
        for key, item in self.hot_cache.items():
            if query.lower() in key.lower():
                results.append({
                    "key": key,
                    "tier": item.tier.value,
                    "size": item.size,
                    "accessed_at": item.accessed_at,
                    "access_count": item.access_count
                })
                
        # Search warm cache
        for key, item in self.warm_cache.items():
            if query.lower() in key.lower():
                results.append({
                    "key": key,
                    "tier": item.tier.value,
                    "size": item.size,
                    "accessed_at": item.accessed_at,
                    "access_count": item.access_count
                })
                
        # Search cold storage
        for key in self.cold_storage:
            if query.lower() in key.lower():
                results.append({
                    "key": key,
                    "tier": MemoryTier.COLD.value,
                    "size": 0,  # Size not immediately available
                    "accessed_at": 0,
                    "access_count": 0
                })
                
        return results[:max_results]
        
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        await self._update_stats()
        
        return {
            "hot_size_mb": self.hot_usage / (1024 * 1024),
            "warm_size_mb": self.warm_usage / (1024 * 1024),
            "cold_size_mb": self.cold_usage / (1024 * 1024),
            "total_items": len(self.hot_cache) + len(self.warm_cache) + len(self.cold_storage),
            "hot_items": len(self.hot_cache),
            "warm_items": len(self.warm_cache),
            "cold_items": len(self.cold_storage),
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "hit_rate": self._calculate_hit_rate(),
            "hot_utilization": (self.hot_usage / self.hot_cache_size) * 100,
            "warm_utilization": (self.warm_usage / self.warm_cache_size) * 100,
        }
        
    def _select_optimal_tier(self, data_size: int, tier_hint: str) -> MemoryTier:
        """Select optimal tier for data storage."""
        if tier_hint != "auto":
            try:
                return MemoryTier(tier_hint)
            except ValueError:
                pass
                
        # Automatic tier selection based on size and usage
        if data_size < 1024:  # < 1KB
            return MemoryTier.HOT
        elif data_size < 1024 * 1024:  # < 1MB
            return MemoryTier.WARM
        else:
            return MemoryTier.COLD
            
    async def _store_hot(self, key: str, item: MemoryItem) -> bool:
        """Store item in hot cache."""
        try:
            # Check if we need to evict items
            while self.hot_usage + item.size > self.hot_cache_size and self.hot_cache:
                await self._evict_hot_item()
                
            # Store item
            self.hot_cache[key] = item
            self.hot_usage += item.size
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing in hot cache: {e}")
            return False
            
    async def _store_warm(self, key: str, item: MemoryItem) -> bool:
        """Store item in warm cache."""
        try:
            # Compress data
            compressed_data = await self._compress_data(item.data)
            compressed_item = MemoryItem(
                key=key,
                data=compressed_data,
                size=len(compressed_data),
                tier=MemoryTier.WARM,
                compressed=True,
                checksum=item.checksum
            )
            
            # Check if we need to evict items
            while self.warm_usage + compressed_item.size > self.warm_cache_size and self.warm_cache:
                await self._evict_warm_item()
                
            # Store item
            self.warm_cache[key] = compressed_item
            self.warm_usage += compressed_item.size
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing in warm cache: {e}")
            return False
            
    async def _store_cold(self, key: str, item: MemoryItem) -> bool:
        """Store item in cold storage."""
        try:
            # Create file path
            file_path = self.cold_storage_path / f"{key}.pkl"
            
            # Serialize and compress data
            serialized_data = pickle.dumps(item.data)
            compressed_data = zlib.compress(serialized_data, self.compression_level)
            
            # Write to file
            with open(file_path, 'wb') as f:
                f.write(compressed_data)
                
            # Update cold storage index
            self.cold_storage[key] = str(file_path)
            self.cold_usage += len(compressed_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing in cold storage: {e}")
            return False
            
    async def _evict_hot_item(self):
        """Evict least recently used item from hot cache."""
        if not self.hot_cache:
            return
            
        # Get least recently used item
        key, item = self.hot_cache.popitem(last=False)
        self.hot_usage -= item.size
        
        # Promote to warm cache
        await self._promote_to_warm(key, item.data)
        
    async def _evict_warm_item(self):
        """Evict least recently used item from warm cache."""
        if not self.warm_cache:
            return
            
        # Find least recently used item
        oldest_key = min(self.warm_cache.keys(), 
                        key=lambda k: self.warm_cache[k].accessed_at)
        
        item = self.warm_cache.pop(oldest_key)
        self.warm_usage -= item.size
        
        # Promote to cold storage
        data = await self._decompress_data(item.data)
        await self._promote_to_cold(oldest_key, data)
        
    async def _promote_to_hot(self, key: str, data: Any):
        """Promote item to hot cache."""
        item = MemoryItem(
            key=key,
            data=data,
            size=len(pickle.dumps(data)),
            tier=MemoryTier.HOT
        )
        
        await self._store_hot(key, item)
        
    async def _promote_to_warm(self, key: str, data: Any):
        """Promote item to warm cache."""
        item = MemoryItem(
            key=key,
            data=data,
            size=len(pickle.dumps(data)),
            tier=MemoryTier.WARM
        )
        
        await self._store_warm(key, item)
        
    async def _promote_to_cold(self, key: str, data: Any):
        """Promote item to cold storage."""
        item = MemoryItem(
            key=key,
            data=data,
            size=len(pickle.dumps(data)),
            tier=MemoryTier.COLD
        )
        
        await self._store_cold(key, item)
        
    async def _compress_data(self, data: Any) -> bytes:
        """Compress data using zlib."""
        serialized_data = pickle.dumps(data)
        return zlib.compress(serialized_data, self.compression_level)
        
    async def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data using zlib."""
        decompressed_data = zlib.decompress(compressed_data)
        return pickle.loads(decompressed_data)
        
    async def _load_cold_data(self, key: str, strategy: LoadingStrategy) -> Optional[Any]:
        """Load data from cold storage."""
        if key not in self.cold_storage:
            return None
            
        file_path = Path(self.cold_storage[key])
        
        if not file_path.exists():
            # Clean up broken reference
            del self.cold_storage[key]
            return None
            
        try:
            if strategy == LoadingStrategy.MMAP:
                return await self._load_mmap_data(key, file_path)
            else:
                # Full loading
                with open(file_path, 'rb') as f:
                    compressed_data = f.read()
                    
                decompressed_data = zlib.decompress(compressed_data)
                return pickle.loads(decompressed_data)
                
        except Exception as e:
            self.logger.error(f"Error loading cold data {key}: {e}")
            return None
            
    async def _load_mmap_data(self, key: str, file_path: Path) -> Optional[Any]:
        """Load data using memory mapping."""
        try:
            with open(file_path, 'rb') as f:
                mmap_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.mmap_files[key] = mmap_file
                
                # Decompress and deserialize
                compressed_data = mmap_file.read()
                decompressed_data = zlib.decompress(compressed_data)
                return pickle.loads(decompressed_data)
                
        except Exception as e:
            self.logger.error(f"Error loading mmap data {key}: {e}")
            return None
            
    async def _remove_item(self, key: str) -> bool:
        """Remove item from all tiers."""
        removed = False
        
        # Remove from hot cache
        if key in self.hot_cache:
            item = self.hot_cache.pop(key)
            self.hot_usage -= item.size
            removed = True
            
        # Remove from warm cache
        if key in self.warm_cache:
            item = self.warm_cache.pop(key)
            self.warm_usage -= item.size
            removed = True
            
        # Remove from cold storage
        if key in self.cold_storage:
            file_path = Path(self.cold_storage.pop(key))
            if file_path.exists():
                file_path.unlink()
                removed = True
                
        # Clean up memory mapping
        if key in self.mmap_files:
            self.mmap_files[key].close()
            del self.mmap_files[key]
            
        return removed
        
    def _get_item_tier(self, key: str) -> Optional[str]:
        """Get the tier of an item."""
        if key in self.hot_cache:
            return MemoryTier.HOT.value
        elif key in self.warm_cache:
            return MemoryTier.WARM.value
        elif key in self.cold_storage:
            return MemoryTier.COLD.value
        return None
        
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.stats.cache_hits + self.stats.cache_misses
        if total_requests == 0:
            return 0.0
        return (self.stats.cache_hits / total_requests) * 100
        
    async def _update_stats(self):
        """Update memory statistics."""
        # Update sizes
        self.hot_usage = sum(item.size for item in self.hot_cache.values())
        self.warm_usage = sum(item.size for item in self.warm_cache.values())
        
        # Update cold storage size
        cold_size = 0
        for file_path in self.cold_storage.values():
            path = Path(file_path)
            if path.exists():
                cold_size += path.stat().st_size
        self.cold_usage = cold_size
        
    async def _memory_cleanup_loop(self):
        """Background memory cleanup loop."""
        while self.is_running:
            try:
                if time.time() - self.last_cleanup > self.cleanup_interval:
                    await self._cleanup_memory_tiers()
                    self.last_cleanup = time.time()
                    
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in memory cleanup loop: {e}")
                await asyncio.sleep(300)
                
    async def _memory_optimization_loop(self):
        """Background memory optimization loop."""
        while self.is_running:
            try:
                await self._optimize_memory_usage()
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in memory optimization loop: {e}")
                await asyncio.sleep(600)
                
    async def _memory_monitoring_loop(self):
        """Background memory monitoring loop."""
        while self.is_running:
            try:
                await self._monitor_memory_health()
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in memory monitoring loop: {e}")
                await asyncio.sleep(60)
                
    async def _cleanup_memory_tiers(self):
        """Clean up memory tiers."""
        # Clean up expired items
        current_time = time.time()
        
        # Clean hot cache
        expired_keys = []
        for key, item in self.hot_cache.items():
            if current_time - item.accessed_at > 3600:  # 1 hour
                expired_keys.append(key)
                
        for key in expired_keys:
            await self._remove_item(key)
            
        # Clean warm cache
        expired_keys = []
        for key, item in self.warm_cache.items():
            if current_time - item.accessed_at > 7200:  # 2 hours
                expired_keys.append(key)
                
        for key in expired_keys:
            await self._remove_item(key)
            
    async def _optimize_memory_usage(self):
        """Optimize memory usage."""
        # Promote frequently accessed items
        for key, item in self.warm_cache.items():
            if item.access_count > 10:
                data = await self._decompress_data(item.data)
                await self._promote_to_hot(key, data)
                
    async def _monitor_memory_health(self):
        """Monitor memory health."""
        # Check for memory pressure
        hot_utilization = (self.hot_usage / self.hot_cache_size) * 100
        warm_utilization = (self.warm_usage / self.warm_cache_size) * 100
        
        if hot_utilization > 90:
            self.logger.warning(f"Hot cache utilization high: {hot_utilization:.1f}%")
            
        if warm_utilization > 90:
            self.logger.warning(f"Warm cache utilization high: {warm_utilization:.1f}%")
            
    async def _compact_memory(self):
        """Compact memory by removing unused items."""
        # Implementation for memory compaction
        pass
        
    async def _load_cold_storage_index(self):
        """Load cold storage index from disk."""
        index_file = self.cold_storage_path / "index.pkl"
        if index_file.exists():
            try:
                with open(index_file, 'rb') as f:
                    self.cold_storage = pickle.load(f)
            except Exception as e:
                self.logger.error(f"Error loading cold storage index: {e}")
                
    async def _save_cold_storage_index(self):
        """Save cold storage index to disk."""
        index_file = self.cold_storage_path / "index.pkl"
        try:
            with open(index_file, 'wb') as f:
                pickle.dump(self.cold_storage, f)
        except Exception as e:
            self.logger.error(f"Error saving cold storage index: {e}")
            
    async def _initialize_memory_tiers(self):
        """Initialize memory tiers."""
        # Any initialization specific to memory tiers
        pass
        
    async def _cleanup_memory_mappings(self):
        """Clean up memory mappings."""
        for mmap_file in self.mmap_files.values():
            mmap_file.close()
        self.mmap_files.clear()
        
    async def _send_response(self, original_message: AgentMessage, response: Dict[str, Any]):
        """Send response to a message."""
        # Implementation depends on message bus integration
        pass
        
    async def _send_error_response(self, original_message: AgentMessage, error: str):
        """Send error response to a message."""
        await self._send_response(original_message, {
            "status": "error",
            "error": error
        })