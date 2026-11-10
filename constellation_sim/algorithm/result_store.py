"""Chunked simulation-result persistence with legacy file compatibility."""

import json
import marshal
import queue
import sqlite3
import threading
import zlib
from pathlib import Path


class ResultStore:
    def __init__(self,path,write=False,chunk_frames=30):
        self.path=Path(path);self.write=write;self.chunk_frames=max(1,int(chunk_frames));self.lock=threading.Lock()
        self.write_queue=None;self.writer_thread=None;self.db=None;self.chunk_db=None;self.chunk_id=None;self.pending=0;self._indices_cache=None
        self.directory_mode=write or self.path.is_dir()
        if self.directory_mode:
            if write:self.path.mkdir(parents=True,exist_ok=True)
            elif not self.path.is_dir():raise FileNotFoundError(self.path)
            self.metadata_path=self.path/"metadata.json"
            if write:
                self.metadata={"storage":{"version":4,"mode":"chunked","chunk_frames":self.chunk_frames},"chunks":[]}
                self._save_metadata();self.write_queue=queue.Queue();self.writer_thread=threading.Thread(target=self._writer_loop,daemon=True);self.writer_thread.start()
            else:
                self.metadata=json.loads(self.metadata_path.read_text(encoding="utf-8"));self.chunk_frames=int(self.metadata.get("storage",{}).get("chunk_frames",30))
        else:
            self.metadata={};self.db=sqlite3.connect(str(self.path),check_same_thread=False);self.db.execute("PRAGMA journal_mode=WAL");self.db.execute("PRAGMA synchronous=NORMAL")

    def _save_metadata(self):
        temporary=self.metadata_path.with_suffix(".tmp");temporary.write_text(json.dumps(self.metadata,ensure_ascii=False,separators=(',',':')),encoding="utf-8");temporary.replace(self.metadata_path)

    def set_metadata(self,key,value):
        if self.directory_mode:
            if key=="offline_plans":
                # 离线方案包含大量逐路由数据。若嵌入metadata，每新增一个
                # 30帧分片都要重新序列化十余MB，周期性占用GIL并卡住界面。
                filename="offline_plans.json";temporary=(self.path/filename).with_suffix(".tmp")
                temporary.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':')),encoding="utf-8");temporary.replace(self.path/filename)
                self.metadata["offline_plans_file"]=filename;self.metadata.pop("offline_plans",None);self._save_metadata();return
            self.metadata[key]=value
            if key=="config":(self.path/"config_snapshot.json").write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
            self._save_metadata();return
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)");self.db.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",(key,json.dumps(value,ensure_ascii=False,separators=(',',':'))));self.db.commit()

    def get_metadata(self,key,default=None):
        if self.directory_mode:
            if key=="offline_plans" and key not in self.metadata:
                path=self.path/self.metadata.get("offline_plans_file","offline_plans.json")
                if path.is_file():return json.loads(path.read_text(encoding="utf-8"))
            return self.metadata.get(key,default)
        row=self.db.execute("SELECT value FROM metadata WHERE key=?",(key,)).fetchone();return json.loads(row[0]) if row else default

    def _chunk_path(self,chunk_id):
        start=chunk_id*self.chunk_frames;end=start+self.chunk_frames-1;return self.path/f"frames_{start:06d}_{end:06d}.hysim"

    @staticmethod
    def _configure_chunk(connection):
        connection.execute("PRAGMA journal_mode=DELETE");connection.execute("PRAGMA synchronous=NORMAL");connection.execute("CREATE TABLE IF NOT EXISTS frames(frame_index INTEGER PRIMARY KEY,time_s REAL NOT NULL,absolute_time TEXT,blob BLOB NOT NULL)");connection.execute("CREATE INDEX IF NOT EXISTS idx_frames_time ON frames(time_s)")

    def _open_write_chunk(self,chunk_id):
        if self.chunk_id==chunk_id:return
        if self.chunk_db is not None:self.chunk_db.commit();self.chunk_db.close()
        self.chunk_id=chunk_id;chunk_path=self._chunk_path(chunk_id);self.chunk_db=sqlite3.connect(str(chunk_path),check_same_thread=False);self._configure_chunk(self.chunk_db)
        if chunk_path.name not in self.metadata["chunks"]:
            self.metadata["chunks"].append(chunk_path.name);self.metadata["chunks"].sort()
            # 每10个分片更新一次索引；结束时_writer_loop会完整写回。
            if len(self.metadata["chunks"])%10==0:self._save_metadata()

    def write_frame(self,frame):
        blob=frame.get("_result_blob")
        self.write_queue.put((frame["index"],frame["time_s"],frame.get("absolute_time",""),blob) if blob is not None else frame)

    def _writer_loop(self):
        while True:
            frame=self.write_queue.get()
            if frame is None:break
            if isinstance(frame,tuple):frame_index,time_s,absolute_time,blob=frame
            else:frame_index,time_s,absolute_time=frame["index"],frame["time_s"],frame.get("absolute_time","");blob=b"M"+zlib.compress(marshal.dumps(frame),1)
            with self.lock:
                self._open_write_chunk(int(frame_index)//self.chunk_frames);self.chunk_db.execute("INSERT OR REPLACE INTO frames(frame_index,time_s,absolute_time,blob) VALUES(?,?,?,?)",(frame_index,time_s,absolute_time,blob));self.pending+=1
                if self.pending>=1:self.chunk_db.commit();self.pending=0
        with self.lock:
            if self.chunk_db is not None:self.chunk_db.commit();self.chunk_db.close();self.chunk_db=None
            self.pending=0;self._save_metadata()

    @staticmethod
    def _decode(blob):
        if blob[:1]==b"M":return marshal.loads(zlib.decompress(blob[1:]))
        return json.loads(zlib.decompress(blob).decode("utf-8"))

    def read_frame(self,index):
        if not self.directory_mode:
            row=self.db.execute("SELECT blob FROM frames WHERE frame_index=?",(index,)).fetchone();return self._decode(row[0]) if row else None
        path=self._chunk_path(int(index)//self.chunk_frames)
        if not path.is_file():return None
        db=sqlite3.connect(str(path));row=db.execute("SELECT blob FROM frames WHERE frame_index=?",(index,)).fetchone();db.close();return self._decode(row[0]) if row else None

    def frame_indices(self):
        if not self.directory_mode:return [row[0] for row in self.db.execute("SELECT frame_index FROM frames ORDER BY frame_index")]
        if self._indices_cache is not None:return list(self._indices_cache)
        indices=[]
        for name in self.metadata.get("chunks",[]):
            path=self.path/name
            if not path.is_file():continue
            db=sqlite3.connect(str(path));indices.extend(row[0] for row in db.execute("SELECT frame_index FROM frames ORDER BY frame_index"));db.close()
        self._indices_cache=sorted(set(indices));return list(self._indices_cache)

    def frame_count(self):return len(self.frame_indices()) if self.directory_mode else self.db.execute("SELECT COUNT(*) FROM frames").fetchone()[0]

    def nearest_frame(self,index):
        indices=self.frame_indices()
        if not indices:return None,None
        actual=min(indices,key=lambda value:abs(value-index));return actual,self.read_frame(actual)

    def close(self):
        if self.writer_thread:self.write_queue.put(None);self.writer_thread.join();self.writer_thread=None
        if self.db is not None:
            with self.lock:self.db.commit();self.db.close();self.db=None
