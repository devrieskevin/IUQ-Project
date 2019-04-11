import socket
import select
import dill

import scheduler

if __name__ == "__main__":
    parser = argeparse.ArgumentParser()
    parser.add_argument("--server_host",dest="server_host",required=True)
    parser.add_argument("--port",dest="port",default=6677)
    parser.add_argument("--nprocs",dest="nprocs",default=16)

    args = parser.parse_args()

    server_host = args.server_host
    port = args.port
    nprocs = args.port

    # INITIALIZE CLIENT

    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.connect((server_host,port))

    print("Connected to server")

    runscheduler = scheduler.ModelScheduler(nprocs)

    # Send client data to server
    data = {}
    data["nprocs"] = nprocs

    message = dill.dumps(data)
    s.send(message)

    indices = []
    runs = []
    while True:
        rlist,wlist,xlist = select.select([s],[s],[])

        if rlist:
            message = s.recv(1024)

            if not message:
                print("Connection with server broken")
            else:
                data = dill.loads(message)

                # Keep track of the indices of the runs on the server
                run_indices = run_indices + data["indices"]

                # Store and enqueue the runs
                runs = runs + data["runs"]
                runscheduler.enqueueBatch(data["runs"])
                runscheduler.pushQueue()

        if wlist:
            message = None
            data = dill.dumps(message)
            s.send(data)

        runscheduler.pushQueue()

    s.close()
