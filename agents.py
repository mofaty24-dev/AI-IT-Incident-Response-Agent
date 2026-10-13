import asyncio
from agents import Agent, Runner
from tools import check_device_status, search_it_knowledge, check_account_status, check_network_status, check_vpn_status

# Specialized Worker Agents
network_worker = Agent(
    name="NetworkSpecialist",
    instructions="Analyze network and VPN connectivity issues using available tools.",
    tools=[check_vpn_status, check_network_status]
)

device_worker = Agent(
    name="DeviceSpecialist",
    instructions="Analyze user device health and performance.",
    tools=[check_device_status, search_it_knowledge]
)

account_worker = Agent(
    name="AccountSpecialist",
    instructions="Analyze Active Directory account issues and security concerns.",
    tools=[check_account_status, search_it_knowledge]
)

# Parallel Diagnostic Function fulfilling the Parallelization requirement
async def run_parallel_diagnostics(employee_id: str):
    """Run network and device checks concurrently using asyncio.gather."""
    net_result, dev_result, acc_result = await asyncio.gather(
        Runner.run(network_worker, f"Check network and VPN for employee {employee_id}"),
        Runner.run(device_worker, f"Check device status for employee {employee_id}"),
        Runner.run(account_worker, f"Check account status for employee {employee_id}")
    )
    return {
        "network_diagnosis": net_result.final_output,
        "device_diagnosis": dev_result.final_output,
        "account_diagnosis": acc_result.final_output
    }
