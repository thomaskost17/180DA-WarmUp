import socket
for i in range(10):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   
    client.connect(('0.0.0.0', 8081))
    st = 'I am CLIENT\n'
    client.send(st.encode())
    from_server = client.recv(4096)
    client.close()
    print(from_server.decode())