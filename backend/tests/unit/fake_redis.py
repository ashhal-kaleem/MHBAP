import time

class FakePipeline:
    def __init__(self, redis):
        self.redis = redis
        self.commands = []

    def zadd(self, key, mapping):
        self.commands.append(("zadd", key, mapping))
        return self

    def expire(self, key, seconds):
        self.commands.append(("expire", key, seconds))
        return self

    def zremrangebyscore(self, key, min_score, max_score):
        self.commands.append(("zremrangebyscore", key, min_score, max_score))
        return self

    def zcard(self, key):
        self.commands.append(("zcard", key))
        return self

    def zrange(self, key, start, end, withscores=False):
        self.commands.append(("zrange", key, start, end, withscores))
        return self

    async def execute(self):
        results = []
        for cmd in self.commands:
            op = cmd[0]
            if op == "zadd":
                res = await self.redis.zadd(cmd[1], cmd[2])
                results.append(res)
            elif op == "expire":
                res = await self.redis.expire(cmd[1], cmd[2])
                results.append(res)
            elif op == "zremrangebyscore":
                res = await self.redis.zremrangebyscore(cmd[1], cmd[2], cmd[3])
                results.append(res)
            elif op == "zcard":
                res = await self.redis.zcard(cmd[1])
                results.append(res)
            elif op == "zrange":
                res = await self.redis.zrange(cmd[1], cmd[2], cmd[3], withscores=cmd[4])
                results.append(res)
        return results

class FakeRedis:
    def __init__(self):
        self.store = {}
        self.zsets = {}

    def pipeline(self):
        return FakePipeline(self)

    async def set(self, key, value, ex=None):
        self.store[key] = (value, time.time() + (ex or 0))

    async def exists(self, key):
        if key in self.store:
            val, exp = self.store[key]
            if exp >= time.time():
                return 1
            else:
                del self.store[key]
        return 0

    async def keys(self, pattern):
        prefix = pattern.replace('*', '')
        return [k for k in self.store.keys() if k.startswith(prefix)]

    async def delete(self, *keys):
        for k in keys:
            self.store.pop(k, None)
            self.zsets.pop(k, None)

    async def expire(self, key, seconds):
        # We don't implement full TTL eviction for fake ZSETs
        pass

    async def zadd(self, key, mapping):
        if key not in self.zsets:
            self.zsets[key] = {}
        for member, score in mapping.items():
            self.zsets[key][member] = score
        return len(mapping)

    async def zremrangebyscore(self, key, min_score, max_score):
        if key not in self.zsets:
            return 0
        to_remove = []
        for member, score in self.zsets[key].items():
            if min_score <= score <= max_score:
                to_remove.append(member)
        for m in to_remove:
            del self.zsets[key][m]
        return len(to_remove)

    async def zcard(self, key):
        if key not in self.zsets:
            return 0
        return len(self.zsets[key])

    async def zrange(self, key, start, end, withscores=False):
        if key not in self.zsets:
            return []
        items = list(self.zsets[key].items())
        items.sort(key=lambda x: x[1])
        slice_end = end + 1 if end != -1 else None
        res_items = items[start:slice_end]
        if withscores:
            return [(m, s) for m, s in res_items]
        return [m for m, s in res_items]

    def clear(self):
        self.store.clear()
        self.zsets.clear()
