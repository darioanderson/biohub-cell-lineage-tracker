"""Read selected public archive members via bounded HTTP ranges."""
import io
import requests


class RangeReader(io.RawIOBase):
    def __init__(self,url,budget=512*1024*1024):
        self.url=url
        self.session=requests.Session()
        self.pos=0
        self.transferred=0
        self.budget=budget
        with self.session.get(url,headers={'Range':'bytes=-22'},stream=True,timeout=(20,90)) as response:
            response.raise_for_status()
            if response.status_code!=206 or '/' not in response.headers.get('Content-Range',''):
                raise ValueError('Byte ranges unavailable; refusing to download the whole archive.')
            self.size=int(response.headers['Content-Range'].split('/')[-1])

    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos

    def read(self,length=-1):
        length=self.size-self.pos if length<0 else min(length,self.size-self.pos)
        if not length:return b''
        if length>32*1024*1024 or self.transferred+length>self.budget:
            raise ValueError('Selected archive read exceeds the bounded transfer budget.')
        with self.session.get(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+length-1}'},stream=True,timeout=(20,90)) as response:
            if response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {self.pos}-'):
                raise ValueError('Server returned a different range; refusing full download.')
            data=response.raw.read(length+1)
            if len(data)!=length:raise ValueError('Truncated archive range.')
        self.pos+=length
        self.transferred+=length
        return data
