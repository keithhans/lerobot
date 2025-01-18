import socket
import pickle
import struct
import torch

from lerobot.common.policies.act.modeling_act import ACTPolicy

def recv_msg(sock):
    # First receive the message length as a 4-byte integer
    raw_msglen = sock.recv(4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    
    # Now receive the message data
    data = bytearray()
    while len(data) < msglen:
        packet = sock.recv(min(msglen - len(data), 4096))
        if not packet:
            return None
        data.extend(packet)
    return data

def send_msg(sock, msg):
    # Prefix message with its length as a 4-byte integer
    msg_length = struct.pack('>I', len(msg))
    sock.sendall(msg_length + msg)

def main():
    # Initialize policy
    device = "cuda"  # TODO: On Mac, use "mps" or "cpu"
    ckpt_path = "outputs/train/act_mycobot_real/checkpoints/last/pretrained_model"
    policy = ACTPolicy.from_pretrained(ckpt_path)
    policy.to(device)

    # Set up server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', 6666))
    server_socket.listen(1)
    print("Server listening on port 6666...")

    while True:
        # Accept client connection
        client_socket, address = server_socket.accept()
        print(f"Connected to client at {address}")

        try:
            while True:
                # Receive observation data
                data = recv_msg(client_socket)
                if data is None:
                    break

                # Deserialize observation
                observation = pickle.loads(data)
                print(f"Received observation with keys: {observation.keys()}")

                # Move observation to device
                for name in observation:
                    if "image" in name:
                        observation[name] = observation[name].type(torch.float32) / 255
                    observation[name] = observation[name].to(device)

                # Compute action
                action = policy.select_action(observation)
                action = action.squeeze(0).to('cpu')
                print("action:", action)

                # Send action back to client
                action_data = pickle.dumps(action)
                send_msg(client_socket, action_data)

        except Exception as e:
            print(f"Error processing client request: {e}")
        finally:
            client_socket.close()

if __name__ == "__main__":
    main() 