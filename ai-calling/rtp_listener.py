import socket

UDP_IP = "0.0.0.0"
UDP_PORT = 9000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print("Listening for RTP audio on port 9000...")

while True:
    packet, addr = sock.recvfrom(2048)

    print("packet received:", len(packet))

    # RTP header = first 12 bytes
    audio = packet[12:]

    print("audio payload:", len(audio))