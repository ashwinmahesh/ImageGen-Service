"""Bounded local IPC, with per-job lock descriptors passed to model descendants."""
import array
import json
import socket
import struct

MAX_MESSAGE=16*1024*1024


def send(channel,message,fds=()):
    payload=json.dumps(message,allow_nan=False).encode()
    if len(payload)>MAX_MESSAGE:raise RuntimeError('Model IPC message too large')
    header=struct.pack('!I',len(payload))
    rights=[(socket.SOL_SOCKET,socket.SCM_RIGHTS,array.array('i',fds))] if fds else []
    sent=channel.sendmsg([header],rights)
    if sent<4:channel.sendall(header[sent:])
    channel.sendall(payload)


def receive(channel):
    header,ancillary,flags,_=channel.recvmsg(4,socket.CMSG_SPACE(16*array.array('i').itemsize))
    if not header:raise EOFError('Model process disconnected')
    while len(header)<4:
        chunk=channel.recv(4-len(header))
        if not chunk:raise EOFError('Truncated model message')
        header+=chunk
    fds=[]
    for level,kind,data in ancillary:
        if level==socket.SOL_SOCKET and kind==socket.SCM_RIGHTS:
            values=array.array('i');values.frombytes(data[:len(data)//values.itemsize*values.itemsize]);fds.extend(values)
    try:
        size=struct.unpack('!I',header)[0]
        if flags & socket.MSG_CTRUNC or size>MAX_MESSAGE:raise RuntimeError('Invalid model message')
        payload=bytearray()
        while len(payload)<size:
            chunk=channel.recv(min(65536,size-len(payload)))
            if not chunk:raise EOFError('Truncated model message')
            payload.extend(chunk)
        return json.loads(payload),fds
    except BaseException:
        import os
        for descriptor in fds:os.close(descriptor)
        raise
