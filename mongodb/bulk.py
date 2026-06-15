"""
MongoDB 批量写入工具
修复 mongodb/bulk.py:112 写入未使用有序=false 的技术债

使用 ordered=False 的优势：
1. 性能提升：无序写入并行执行，吞吐量提升 2-5 倍
2. 容错性：单条文档失败不中断整个批次
3. 适用场景：5G 上报的时序数据、告警日志、批量导入等
"""

import logging
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import BulkWriteError, PyMongoError
import time

logger = logging.getLogger(__name__)


class BulkWriteResult:
    """批量写入结果封装"""

    def __init__(self):
        self.inserted_count = 0
        self.matched_count = 0
        self.modified_count = 0
        self.upserted_count = 0
        self.deleted_count = 0
        self.errors: List[Dict] = []
        self.execution_time_ms = 0
        self.total_attempted = 0

    @property
    def success_count(self) -> int:
        return (
            self.inserted_count
            + self.modified_count
            + self.upserted_count
            + self.deleted_count
        )

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def to_dict(self) -> Dict:
        return {
            'inserted_count': self.inserted_count,
            'matched_count': self.matched_count,
            'modified_count': self.modified_count,
            'upserted_count': self.upserted_count,
            'deleted_count': self.deleted_count,
            'error_count': self.error_count,
            'success_count': self.success_count,
            'total_attempted': self.total_attempted,
            'execution_time_ms': self.execution_time_ms,
            'errors': self.errors[:50],  # 最多返回前50个错误
            'has_errors': self.has_errors
        }

    def merge(self, other: 'BulkWriteResult'):
        """合并另一个结果"""
        self.inserted_count += other.inserted_count
        self.matched_count += other.matched_count
        self.modified_count += other.modified_count
        self.upserted_count += other.upserted_count
        self.deleted_count += other.deleted_count
        self.errors.extend(other.errors)
        self.execution_time_ms += other.execution_time_ms
        self.total_attempted += other.total_attempted


class BulkWriter:
    """
    MongoDB 批量写入工具
    核心特性：
    - 默认使用 ordered=False（无序写入）提升吞吐量
    - 自动分批次（默认 1000 条/批，MongoDB 限制）
    - 支持失败重试（指数退避）
    - 详细的错误收集
    - 写入前/后钩子
    """

    DEFAULT_BATCH_SIZE = 1000
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 0.5  # 秒
    RETRY_MAX_DELAY = 5.0

    def __init__(
        self,
        mongo_client: Optional[MongoClient] = None,
        db_name: Optional[str] = None,
        ordered: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        bypass_document_validation: bool = False
    ):
        """
        初始化批量写入器
        :param ordered: 是否有序写入（默认 False，无序写入性能更高）
        :param batch_size: 每批文档数（MongoDB 限制单批最多 1000 条）
        """
        if mongo_client and db_name:
            self.db = mongo_client[db_name]
        else:
            # 延迟初始化，使用 Django settings
            from django.conf import settings
            client = MongoClient(
                settings.MONGODB_DATABASES['default']['host'],
                settings.MONGODB_DATABASES['default']['port']
            )
            self.db = client[settings.MONGODB_DATABASES['default']['name']]

        self.ordered = ordered
        self.batch_size = min(max(batch_size, 1), self.DEFAULT_BATCH_SIZE)
        self.bypass_document_validation = bypass_document_validation

        # 统计信息
        self.stats = {
            'total_batches': 0,
            'total_documents': 0,
            'total_errors': 0,
            'total_retries': 0
        }

    def get_collection(self, collection_name: str):
        """获取集合引用"""
        return self.db[collection_name]

    def _chunk_docs(self, docs: List[Dict]) -> List[List[Dict]]:
        """将文档列表分块"""
        return [
            docs[i:i + self.batch_size]
            for i in range(0, len(docs), self.batch_size)
        ]

    def _parse_bulk_error(self, error: BulkWriteError) -> List[Dict]:
        """解析批量写入错误，提取详细信息"""
        parsed_errors = []
        raw_errors = error.details.get('writeErrors', []) if error.details else []

        for err in raw_errors:
            parsed_errors.append({
                'index': err.get('index'),
                'code': err.get('code'),
                'code_name': err.get('codeName'),
                'message': err.get('errmsg'),
                'operation': str(err.get('op', {}))[:200]
            })

        # 写入关注错误
        wc_error = error.details.get('writeConcernError') if error.details else None
        if wc_error:
            parsed_errors.append({
                'type': 'write_concern',
                'code': wc_error.get('code'),
                'message': wc_error.get('errmsg')
            })

        return parsed_errors

    def _parse_bulk_result(self, raw_result) -> BulkWriteResult:
        """解析 PyMongo 批量结果为统一格式"""
        result = BulkWriteResult()

        if hasattr(raw_result, 'bulk_api_result'):
            res = raw_result.bulk_api_result
            result.inserted_count = res.get('nInserted', 0)
            result.matched_count = res.get('nMatched', 0)
            result.modified_count = res.get('nModified', 0)
            result.upserted_count = res.get('nUpserted', 0)
            result.deleted_count = res.get('nRemoved', 0)
        elif hasattr(raw_result, 'inserted_count'):
            # insert_many 返回 InsertManyResult
            result.inserted_count = getattr(raw_result, 'inserted_count', 0)

        return result

    def _execute_with_retry(
        self,
        collection,
        operation: Callable,
        *args,
        **kwargs
    ) -> BulkWriteResult:
        """
        执行写入操作，支持失败重试
        """
        result = BulkWriteResult()
        last_exception = None

        for attempt in range(self.MAX_RETRIES):
            try:
                start_time = time.time()
                raw_result = operation(*args, **kwargs)
                result.merge(self._parse_bulk_result(raw_result))
                result.execution_time_ms = int((time.time() - start_time) * 1000)
                return result

            except BulkWriteError as e:
                result.errors.extend(self._parse_bulk_error(e))
                last_exception = e

                # 部分成功也要计入
                if hasattr(e, 'details') and e.details:
                    result.inserted_count += e.details.get('nInserted', 0)
                    result.matched_count += e.details.get('nMatched', 0)
                    result.modified_count += e.details.get('nModified', 0)
                    result.upserted_count += e.details.get('nUpserted', 0)
                    result.deleted_count += e.details.get('nRemoved', 0)

                # 如果是有序写入，直接返回（已中断）
                if self.ordered:
                    logger.error(f"有序批量写入失败: {e}")
                    return result

            except PyMongoError as e:
                last_exception = e
                logger.warning(f"批量写入异常 (尝试 {attempt+1}/{self.MAX_RETRIES}): {e}")

            # 重试延迟（指数退避 + 抖动）
            if attempt < self.MAX_RETRIES - 1:
                delay = min(
                    self.RETRY_BASE_DELAY * (2 ** attempt),
                    self.RETRY_MAX_DELAY
                )
                jitter = delay * (0.5 - time.time() % 1)
                time.sleep(max(0, delay + jitter))
                self.stats['total_retries'] += 1

        # 重试全部失败
        if last_exception:
            result.errors.append({
                'type': 'fatal',
                'message': f"经过 {self.MAX_RETRIES} 次重试仍失败: {str(last_exception)}"
            })

        return result

    def insert_many(
        self,
        collection_name: str,
        documents: List[Dict],
        ordered: Optional[bool] = None,
        before_insert_hook: Optional[Callable[[List[Dict]], List[Dict]]] = None,
        after_insert_hook: Optional[Callable[[List[Dict], BulkWriteResult], None]] = None
    ) -> BulkWriteResult:
        """
        批量插入文档（默认无序，修复技术债 #3）
        :param collection_name: 集合名
        :param documents: 文档列表
        :param ordered: 是否有序（None 使用全局配置）
        :param before_insert_hook: 插入前钩子，处理文档
        :param after_insert_hook: 插入后钩子
        """
        if not documents:
            return BulkWriteResult()

        use_ordered = ordered if ordered is not None else self.ordered
        collection = self.get_collection(collection_name)
        batches = self._chunk_docs(documents)

        total_result = BulkWriteResult()
        total_result.total_attempted = len(documents)

        for batch_idx, batch in enumerate(batches):
            self.stats['total_batches'] += 1
            self.stats['total_documents'] += len(batch)

            # 插入前钩子
            if before_insert_hook:
                try:
                    batch = before_insert_hook(batch)
                except Exception as e:
                    logger.error(f"插入前钩子执行失败: {e}")
                    total_result.errors.append({
                        'batch': batch_idx,
                        'type': 'hook_error',
                        'message': str(e)
                    })
                    continue

            # 执行插入（关键修复：ordered=False）
            logger.debug(
                f"批量插入 {collection_name}: {len(batch)} 条 "
                f"(批次 {batch_idx+1}/{len(batches)}, ordered={use_ordered})"
            )

            result = self._execute_with_retry(
                collection,
                collection.insert_many,
                batch,
                ordered=use_ordered,
                bypass_document_validation=self.bypass_document_validation
            )

            total_result.merge(result)

            # 插入后钩子
            if after_insert_hook:
                try:
                    after_insert_hook(batch, result)
                except Exception as e:
                    logger.error(f"插入后钩子执行失败: {e}")

            # 错误统计
            if result.has_errors:
                self.stats['total_errors'] += result.error_count
                logger.warning(
                    f"批次 {batch_idx+1} 写入错误: {result.error_count} 条, "
                    f"成功: {result.success_count} 条"
                )

        logger.info(
            f"批量插入完成 {collection_name}: "
            f"总 {total_result.total_attempted}, "
            f"成功 {total_result.success_count}, "
            f"错误 {total_result.error_count}, "
            f"耗时 {total_result.execution_time_ms}ms"
        )

        return total_result

    def bulk_write(
        self,
        collection_name: str,
        operations: List[Any],
        ordered: Optional[bool] = None
    ) -> BulkWriteResult:
        """
        执行通用批量写入操作（混合 insert_one/update_one/update_many/delete_one 等）
        :param operations: pymongo 写入操作列表
        """
        if not operations:
            return BulkWriteResult()

        use_ordered = ordered if ordered is not None else self.ordered
        collection = self.get_collection(collection_name)
        batches = self._chunk_docs(operations)

        total_result = BulkWriteResult()
        total_result.total_attempted = len(operations)

        for batch_idx, batch in enumerate(batches):
            self.stats['total_batches'] += 1
            self.stats['total_documents'] += len(batch)

            logger.debug(
                f"批量写入 {collection_name}: {len(batch)} 条操作 "
                f"(批次 {batch_idx+1}/{len(batches)}, ordered={use_ordered})"
            )

            result = self._execute_with_retry(
                collection,
                collection.bulk_write,
                batch,
                ordered=use_ordered,
                bypass_document_validation=self.bypass_document_validation
            )

            total_result.merge(result)

            if result.has_errors:
                self.stats['total_errors'] += result.error_count

        return total_result

    def update_many(
        self,
        collection_name: str,
        updates: List[Dict],
        ordered: Optional[bool] = None
    ) -> BulkWriteResult:
        """
        批量更新
        :param updates: [{'filter': {...}, 'update': {...}, 'upsert': False}, ...]
        """
        from pymongo import UpdateOne

        operations = []
        for upd in updates:
            operations.append(UpdateOne(
                filter=upd.get('filter', {}),
                update=upd.get('update', {}),
                upsert=upd.get('upsert', False)
            ))

        return self.bulk_write(collection_name, operations, ordered)

    def upsert_many(
        self,
        collection_name: str,
        documents: List[Dict],
        key_fields: List[str],
        ordered: Optional[bool] = None
    ) -> BulkWriteResult:
        """
        批量 upsert（按 key_fields 去重更新或插入）
        :param key_fields: 用于构建查询条件的字段
        """
        from pymongo import UpdateOne

        operations = []
        for doc in documents:
            filter_dict = {k: doc.get(k) for k in key_fields if k in doc}
            if not filter_dict:
                continue
            operations.append(UpdateOne(
                filter=filter_dict,
                update={'$set': doc},
                upsert=True
            ))

        return self.bulk_write(collection_name, operations, ordered)

    def get_stats(self) -> Dict:
        """获取批量写入器统计信息"""
        return {
            **self.stats,
            'ordered': self.ordered,
            'batch_size': self.batch_size,
            'bypass_validation': self.bypass_document_validation
        }


def get_bulk_writer(ordered: bool = False, **kwargs) -> BulkWriter:
    """
    工厂函数：获取默认配置的批量写入器
    默认使用 ordered=False（无序写入，修复技术债 #3）
    """
    return BulkWriter(ordered=ordered, **kwargs)


def batch_insert_with_timestamps(
    collection_name: str,
    documents: List[Dict],
    timestamp_field: str = 'created_at',
    ordered: bool = False,
    **kwargs
) -> BulkWriteResult:
    """
    便捷函数：批量插入并自动添加时间戳
    用于 5G 上报的时序数据、光谱数据等
    """
    def add_timestamp(batch: List[Dict]) -> List[Dict]:
        now = datetime.utcnow()
        for doc in batch:
            if timestamp_field not in doc:
                doc[timestamp_field] = now
        return batch

    writer = get_bulk_writer(ordered=ordered, **kwargs)
    return writer.insert_many(
        collection_name,
        documents,
        before_insert_hook=add_timestamp
    )


# 性能对比基准
if __name__ == '__main__':
    """
    测试 ordered=True vs ordered=False 的性能差异
    运行: python mongodb/bulk.py
    """
    import random
    import string

    def generate_docs(n: int) -> List[Dict]:
        docs = []
        for i in range(n):
            docs.append({
                'test_id': i,
                'value': random.random(),
                'name': ''.join(random.choices(string.ascii_letters, k=8)),
                'data': {
                    'a': random.randint(1, 100),
                    'b': random.random()
                }
            })
        return docs

    docs = generate_docs(5000)
    client = MongoClient('localhost', 27017)
    db = client['test_db']

    # 清理
    db['test_ordered'].drop()
    db['test_unordered'].drop()

    # 有序写入测试
    writer_ordered = BulkWriter(client, 'test_db', ordered=True)
    start = time.time()
    res_ordered = writer_ordered.insert_many('test_ordered', docs)
    time_ordered = time.time() - start

    # 无序写入测试
    writer_unordered = BulkWriter(client, 'test_db', ordered=False)
    start = time.time()
    res_unordered = writer_unordered.insert_many('test_unordered', docs)
    time_unordered = time.time() - start

    print("=" * 60)
    print("MongoDB 批量写入性能对比 (5000条文档)")
    print("=" * 60)
    print(f"ordered=True  (有序):   {time_ordered:.3f}s, "
          f"插入 {res_ordered.inserted_count} 条")
    print(f"ordered=False (无序):   {time_unordered:.3f}s, "
          f"插入 {res_unordered.inserted_count} 条")
    print(f"性能提升:                {((time_ordered/time_unordered)-1)*100:.1f}%")
    print(f"吞吐量提升:              {(len(docs)/time_unordered)/(len(docs)/time_ordered):.2f}x")
    print("=" * 60)

    # 清理
    db['test_ordered'].drop()
    db['test_unordered'].drop()
    client.close()
