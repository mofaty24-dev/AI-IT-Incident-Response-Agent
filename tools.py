from agents import function_tool
from pydantic import BaseModel, Field

class IncidentTriage(BaseModel):
    severity: str = Field(description="Low, Medium, High, or Critical")
    category: str = Field(description="Network, VPN, Account, Device, or Security")
    summary: str = Field(description="Concise summary of the employee's issue")
    is_security_incident: bool = Field(description="True if credential breach, malware, or unauthorized access")

@function_tool
async def check_vpn_status(employee_id: str) -> str:
    """Check the VPN gateway connection and session status for an employee."""
    # Simulated backend system check
    return f"VPN Status for {employee_id}: Active session on gateway-eu-01, packet loss: 0%."

@function_tool
async def check_network_status(employee_id: str) -> str:
    """Check local network connectivity and DNS resolution."""
    return f"Network Status for {employee_id}: Ping to internal gateway successful (12ms)."

@function_tool
async def check_device_status(employee_id: str) -> str:
    """Check device health, disk space, and security endpoint status."""
    return f"Device Status for {employee_id}: EDR agent healthy, disk space 45% free."

@function_tool
async def check_account_status(employee_id: str) -> str:
    """Check Active Directory account lockouts and MFA status."""
    return f"Account Status for {employee_id}: Active, MFA registered, no lockouts."

@function_tool
async def search_it_knowledge(query: str) -> str:
    """Search internal IT knowledge base for troubleshooting runbooks."""
    return f"KB Article: For VPN auth failures, clear DNS cache and re-authenticate SSO."

@function_tool
async def create_jira_ticket(employee_id: str, summary: str, resolution: str) -> str:
    """Create a resolved IT support ticket in Jira."""
    return f"Ticket JIRA-9921 created successfully for {employee_id}: {summary} [Resolved: {resolution}]"