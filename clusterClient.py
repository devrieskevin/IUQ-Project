import os
import socket
import select
import time

import argparse

import dill

import scheduler
import hemocell.model as hemocell

from lisa_config import *
#from cartesius_config import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_host",dest="server_host",required=True)
    parser.add_argument("--port",dest="port",type=int,default=6677)
    parser.add_argument("--nprocs",dest="nprocs",type=int,default=16)

    args = parser.parse_args()

    # Set variables
    server_host = args.server_host
    port = args.port
    nprocs = args.nprocs

    # Setup directory
    os.makedirs("client_output",exist_ok=True)
    os.chdir("client_output")

    # INITIALIZE CLIENT
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    while True:
        try:
            s.connect((server_host,port))
            break
        except socket.error as err:
            print(err)
            print("Will try to connect again in 5 seconds")
            time.sleep(5)

    print("Successfully connected to server")

    runscheduler = scheduler.ModelScheduler(nprocs)

    # Send client data to server
    data = {}
    data["nprocs"] = nprocs

    message = dill.dumps(data)
    message += b"\r\n\r\n\r\n"
    _,wlist,_ = select.select([],[s],[])
    s.sendall(message)

    runs = [None for n in range(nprocs)]

    buf = b''
    messages = []

    runClient = True
    while runClient:
        rlist,_,_ = select.select([s],[],[])

        # Receive bytes and add to buffer
        m = s.recv(1024)
        buf = buf + m

        # Split buffer into messages by delimiter
        split_messages = buf.split(b"\r\n\r\n\r\n")

        # Add messages to list and set rest in buffer
        messages = messages + split_messages[:-1]
        buf = split_messages[-1]

        while messages:
            message = messages.pop(0)

            if not message:
                print("Connection with server broken")
                break
            else:
                data = dill.loads(message)
                command = data["command"]

                if command == "run":
                    for n,run in data["runs"]:
                        runs[n] = run
                        runscheduler.enqueue(run)

                elif command == "refreshBuffer":
                    runscheduler.refreshBuffer()

                    outputs = []
                    for n,run in enumerate(runs):
                        if run is not None and run.output is not None:
                            outputs.append((n,run.output))
                            runs[n] = None

                    message = dill.dumps(outputs)
                    message += b"\r\n\r\n\r\n"
                    _,wlist,_ = select.select([],[s],[])
                    s.sendall(message)

                elif command == "wait":
                    runscheduler.wait()
            
                elif command == "close":
                    runClient = False
                    break

        runscheduler.pushQueue()

    s.close()
    print("Client closed. Goodbye :)")

