import torch
from autonomous_vehicles.base import Message

class CommunicationModule:
    """
    Handles inter-agent message passing within the simulation.
    Ensures messages are only delivered to agents within the sender's comm_range.
    """
    def __init__(self, comm_range: float = 5000.0):
        self.comm_range = comm_range

    def broadcast(self, sender_vehicle, message: Message, state_tensor: torch.Tensor, all_vehicles: list):
        """
        Broadcasts a message to all vehicles within the specified comm_range.
        
        Args:
            sender_vehicle: The vehicle sending the message.
            message: The Message namedtuple.
            state_tensor: The global state tensor for distance math.
            all_vehicles: A list of all vehicle objects to deliver messages to.
        """
        sender_pos = state_tensor[sender_vehicle.agent_index, sender_vehicle.COL_POSITION].unsqueeze(0)
        
        # Calculate distances to all agents in the fleet
        all_pos = state_tensor[:, sender_vehicle.COL_POSITION]
        distances = torch.cdist(sender_pos, all_pos).squeeze(0)
        
        # Determine who is in range
        in_range_mask = (distances <= sender_vehicle.comm_range)
        
        # Deliver to those in range
        for idx, in_range in enumerate(in_range_mask):
            if in_range and idx != sender_vehicle.agent_index:
                # Deliver to the target vehicle's inbox
                target_vehicle = all_vehicles[idx]
                target_vehicle.receive_message(message)

    def process_queue(self, inbox: list, perception_memory: dict):
        """
        Parses raw messages in the inbox into structured tactical data.
        Updates the agent's perception memory.
        """
        for msg in inbox:
            # If it's a threat alert, map it to tactical memory
            if msg.msg_type in ["THREAT_ALERT", "DISTRESS_MAYDAY"]:
                perception_memory[f"TACTICAL_{msg.sender_id}"] = {
                    "position": msg.position,
                    "type": msg.msg_type,
                    "timestamp": msg.timestamp
                }
        
        # Clear inbox after processing
        inbox.clear()