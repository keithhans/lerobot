import socket
import pickle
from lerobot.common.policies.act.modeling_act import ACTPolicy

def main():
    # Initialize policy
    device = "cuda"  # TODO: On Mac, use "mps" or "cpu"
    ckpt_path = "outputs/train/act_mycobot_real/checkpoints/last/pretrained_model"
    policy = ACTPolicy.from_pretrained(ckpt_path)
    policy.to(device)

    # Set up server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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
                data = b""
                while True:
                    packet = client_socket.recv(4096)
                    if not packet:
                        break
                    data += packet
                    if len(packet) < 4096:
                        break
                
                if not data:
                    break

                # Deserialize observation
                observation = pickle.loads(data)
                print("received:", observation.keys())

                # Move observation to device
                for name in observation:
                    observation[name] = observation[name].to(device)

                # Compute action
                action = policy.select_action(observation)
                action = action.squeeze(0).to('cpu')
                print("action:", action)

                # Send action back to client
                client_socket.sendall(pickle.dumps(action))

        except Exception as e:
            print(f"Error processing client request: {e}")
        finally:
            client_socket.close()

if __name__ == "__main__":
    main() 