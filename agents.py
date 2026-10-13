import asyncio
from agents import Agent, Runner

# Specialized Worker Agents
network_worker = Agent(
    name="NetworkSpecialist",
    instructions="Analyze network and VPN connectivity issues using available tools.",
    tools=[check_vpn_status, check_network_status]
)

device_account_worker = Agent(
    name="DeviceAccountSpecialist",
    instructions="Analyze user device health and Active Directory account status.",
    tools=[check_device_status, check_account_status, search_it_knowledge]
)

# Parallel Diagnostic Function fulfilling the Parallelization requirement
async def run_parallel_diagnostics(employee_id: str):
    """Run network and device checks concurrently using asyncio.gather."""
    net_result, dev_result = await asyncio.gather(
        Runner.run(network_worker, f"Check network and VPN for employee {employee_id}"),
        Runner.run(device_account_worker, f"Check device and account status for employee {employee_id}")
    )
    return {
        "network_diagnosis": net_result.final_output,
        "device_diagnosis": dev_result.final_output
    }