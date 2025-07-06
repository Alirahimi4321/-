#!/usr/bin/env python3
"""
Project AWARENESS - Main CLI Entry Point
A fully autonomous, on-device Personal AI Agent for Termux/Android
"""

import asyncio
import sys
import signal
import traceback
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner

from core.awareness import AwarenessKernel
from core.config import AwarenessConfig
from core.logger import setup_logger

# Global instances
console = Console()
logger = setup_logger(__name__)
kernel: Optional[AwarenessKernel] = None

app = typer.Typer(
    name="awareness",
    help="Project AWARENESS - Autonomous AI Agent System",
    add_completion=False,
)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    console.print("\n[yellow]Received shutdown signal. Initiating graceful shutdown...[/yellow]")
    if kernel:
        asyncio.create_task(kernel.shutdown())
    sys.exit(0)


@app.command()
def start(
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
    interactive: bool = typer.Option(True, "--interactive", "-i", help="Start in interactive mode"),
):
    """Start the AWARENESS system."""
    global kernel
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Display banner
    console.print(Panel.fit(
        Text("Project AWARENESS", style="bold cyan", justify="center") + "\n" +
        Text("Autonomous AI Agent System", style="italic", justify="center"),
        border_style="cyan"
    ))
    
    try:
        # Load configuration
        config = AwarenessConfig.load(config_file)
        if debug:
            config.debug = True
            
        # Initialize kernel
        with console.status("[bold green]Initializing AWARENESS kernel..."):
            kernel = AwarenessKernel(config)
            
        # Start the system
        if interactive:
            asyncio.run(interactive_mode(kernel))
        else:
            asyncio.run(daemon_mode(kernel))
            
    except Exception as e:
        logger.error(f"Failed to start AWARENESS: {e}")
        console.print(f"[red]Error: {e}[/red]")
        if debug:
            console.print(traceback.format_exc())
        sys.exit(1)


async def interactive_mode(kernel: AwarenessKernel):
    """Run in interactive CLI mode."""
    console.print("[green]Starting interactive mode...[/green]")
    
    # Initialize kernel
    await kernel.initialize()
    
    # Start background tasks
    kernel_task = asyncio.create_task(kernel.run())
    
    try:
        console.print("[bold green]AWARENESS is now active![/bold green]")
        console.print("Type 'help' for commands, 'exit' to quit.\n")
        
        while True:
            try:
                # Get user input
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: Prompt.ask("[bold blue]awareness>[/bold blue]")
                )
                
                if user_input.lower() in ['exit', 'quit']:
                    break
                elif user_input.lower() == 'help':
                    show_help()
                elif user_input.lower() == 'status':
                    await show_status(kernel)
                elif user_input.lower() == 'clear':
                    console.clear()
                elif user_input.strip():
                    # Send command to kernel
                    response = await kernel.process_command(user_input)
                    console.print(f"[green]{response}[/green]")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                
    finally:
        console.print("[yellow]Shutting down...[/yellow]")
        kernel_task.cancel()
        await kernel.shutdown()


async def daemon_mode(kernel: AwarenessKernel):
    """Run in daemon mode."""
    console.print("[green]Starting daemon mode...[/green]")
    
    # Initialize and run kernel
    await kernel.initialize()
    await kernel.run()


def show_help():
    """Display help information."""
    help_text = """
[bold]Available Commands:[/bold]
  help     - Show this help message
  status   - Display system status
  clear    - Clear the screen
  exit     - Exit the program
  
[bold]Agent Commands:[/bold]
  Any other input will be processed by the AI agent system.
"""
    console.print(Panel(help_text, title="Help", border_style="blue"))


async def show_status(kernel: AwarenessKernel):
    """Display system status."""
    status = await kernel.get_status()
    
    status_text = f"""
[bold]System Status:[/bold]
  Uptime: {status.get('uptime', 'Unknown')}
  Active Agents: {status.get('active_agents', 0)}
  Memory Usage: {status.get('memory_usage', 'Unknown')}
  CPU Usage: {status.get('cpu_usage', 'Unknown')}
  Trust Score: {status.get('trust_score', 'Unknown')}
"""
    
    console.print(Panel(status_text, title="Status", border_style="green"))


@app.command()
def version():
    """Show version information."""
    console.print("[bold]Project AWARENESS v1.0.0[/bold]")
    console.print("Autonomous AI Agent System")


@app.command()
def config(
    create: bool = typer.Option(False, "--create", help="Create default configuration"),
    path: str = typer.Option("config/awareness.toml", "--path", help="Configuration file path"),
):
    """Manage configuration."""
    if create:
        config = AwarenessConfig.create_default()
        config.save(path)
        console.print(f"[green]Created default configuration at {path}[/green]")
    else:
        console.print(f"[blue]Configuration file: {path}[/blue]")


if __name__ == "__main__":
    app()