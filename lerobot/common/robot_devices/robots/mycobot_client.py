import socket
import json
import time
from typing import Optional, List, Union

class MyCobotClient:
    def __init__(self, host='localhost', port=6789):
        """Initialize MyCobot client
        
        Args:
            host: Server hostname/IP
            port: Server port number
        """
        self.host = host
        self.port = port
        self.socket = None
        
    def connect(self):
        """Connect to the joystick server"""
        if self.socket:
            self.disconnect()
            
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        
    def disconnect(self):
        """Disconnect from the server"""
        if self.socket:
            self.socket.close()
            self.socket = None
            
    def _send_command(self, command: str, params: dict = None) -> Union[None, float, List[float]]:
        """Send command to server and get response
        
        Args:
            command: Command string to send
            params: Optional parameters for the command
            
        Returns:
            Server response data
            
        Raises:
            RuntimeError: If not connected or server error
        """
        if not self.socket:
            raise RuntimeError("Not connected to server")
            
        try:
            # Send request
            request = {
                'command': command,
                'params': params or {}
            }
            request_data = json.dumps(request).encode('utf-8')
            self.socket.sendall(request_data)
            
            # Get response
            response = self.socket.recv(1024).decode('utf-8')
            result = json.loads(response)
            
            if result['status'] != 'ok':
                raise RuntimeError(f"Server error: {result.get('error', 'unknown error')}")
                
            return result['data']
            
        except Exception as e:
            self.disconnect()  # Reset connection on error
            raise RuntimeError(f"Communication error: {e}")
        
    def get_gripper_value(self, use_robot_data: bool = False) -> float:
        """Get current gripper value
        
        Args:
            use_robot_data: If True, get value directly from robot, otherwise use cached value
        """
        return self._send_command('get_gripper_value', {'use_robot_data': use_robot_data})
        
    def get_action(self) -> Optional[List[float]]:
        """Get current robot action (position + gripper)
        
        Returns:
            List of [x,y,z,rx,ry,rz,gripper] or None if not available
        """
        return self._send_command('get_action')
        
    def get_coords(self, use_robot_data: bool = False) -> Optional[List[float]]:
        """Get current coordinates
        
        Args:
            use_robot_data: If True, get coords directly from robot, otherwise use cached value
            
        Returns:
            List of [x,y,z,rx,ry,rz] or None if not available
        """
        return self._send_command('get_coords', {'use_robot_data': use_robot_data})

def main():
    """Example usage of the client"""
    client = MyCobotClient()
    
    try:
        print("Connecting to joystick server...")
        client.connect()
        
        # Monitor robot state
        while True:
            try:
                action = client.get_action()
                if action is not None:
                    print(f"Current action: {action}")
                    
                gripper = client.get_gripper_value(False)
                print(f"Gripper value: {gripper}")
                
                coords = client.get_coords(False)
                print(f"Current coords: {coords}")
                
                time.sleep(5)  # Poll at 10Hz
                
            except Exception as e:
                print(f"Error getting robot state: {e}")
                time.sleep(1)  # Wait before retry
                try:
                    client.connect()  # Try to reconnect
                except:
                    pass
                
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        client.disconnect()

if __name__ == "__main__":
    main() 