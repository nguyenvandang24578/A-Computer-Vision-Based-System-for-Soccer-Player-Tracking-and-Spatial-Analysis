import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import os
import argparse
from collections import defaultdict
import pandas as pd


class HeatmapGenerator:
    """
    Generate heatmaps and visualizations from tracked position data
    """
    
    def __init__(self, field_dimensions: Tuple[int, int] = (400, 300)):
        """
        Initialize heatmap generator
        
        Args:
            field_dimensions: Size of the field map (width, height)
        """
        self.field_width, self.field_height = field_dimensions
        
        # Colors for teams
        self.team_colors = {
            0: '#FF6464',  # Team 0 - Red
            1: '#64FF64',  # Team 1 - Green
            'ball': '#FFD700',  # Ball - Gold
        }
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def load_position_data(self, file_path: str) -> Dict[str, Any]:
        """Load position data from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def create_player_heatmap(self, positions: List[Tuple[float, float, int]], 
                            title: str = "Player Heatmap", 
                            grid_size: int = 50) -> np.ndarray:
        """
        Create heatmap for a single player
        
        Args:
            positions: List of (x, y, frame) tuples
            title: Title for the heatmap
            grid_size: Resolution of the heatmap grid
            
        Returns:
            Heatmap as numpy array
        """
        if not positions:
            return np.zeros((grid_size, grid_size))
        
        # Extract x, y coordinates
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        # Create 2D histogram
        heatmap, x_edges, y_edges = np.histogram2d(
            x_coords, y_coords, 
            bins=grid_size,
            range=[[0, self.field_width], [0, self.field_height]]
        )
        
        return heatmap.T  # Transpose for correct orientation
    
    def create_team_heatmap(self, data: Dict[str, Any], team_id: int, 
                          grid_size: int = 50) -> np.ndarray:
        """
        Create combined heatmap for all players in a team
        
        Args:
            data: Position data dictionary
            team_id: Team ID (0 or 1)
            grid_size: Resolution of the heatmap grid
            
        Returns:
            Combined team heatmap
        """
        combined_heatmap = np.zeros((grid_size, grid_size))
        player_count = 0
        
        for jersey_number, player_data in data['players'].items():
            if player_data['team'] == team_id:
                positions = player_data['positions']
                if positions:
                    player_heatmap = self.create_player_heatmap(positions, grid_size=grid_size)
                    combined_heatmap += player_heatmap
                    player_count += 1
        
        # Normalize by number of players
        if player_count > 0:
            combined_heatmap /= player_count
        
        return combined_heatmap
    
    def plot_player_heatmap(self, data: Dict[str, Any], jersey_number: str, 
                          output_path: str = None, show: bool = True):
        """
        Plot and save individual player heatmap
        
        Args:
            data: Position data dictionary
            jersey_number: Player's jersey number
            output_path: Path to save the plot
            show: Whether to display the plot
        """
        if jersey_number not in data['players']:
            print(f"Player {jersey_number} not found in data")
            return
        
        player_data = data['players'][jersey_number]
        positions = player_data['positions']
        team_id = player_data['team']
        
        if not positions:
            print(f"No position data for player {jersey_number}")
            return
        
        # Create heatmap
        heatmap = self.create_player_heatmap(positions)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot heatmap
        im = ax.imshow(heatmap, cmap='hot', interpolation='bilinear', 
                      extent=[0, self.field_width, 0, self.field_height],
                      origin='lower', alpha=0.7)
        
        # Add field lines (simplified)
        self._draw_field_lines(ax)
        
        # Customize plot
        ax.set_title(f'Player {jersey_number} Heatmap (Team {team_id})', fontsize=16)
        ax.set_xlabel('Field Width (pixels)', fontsize=12)
        ax.set_ylabel('Field Height (pixels)', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Position Density', fontsize=12)
        
        # Add statistics
        total_positions = len(positions)
        frames_range = f"{min(pos[2] for pos in positions)} - {max(pos[2] for pos in positions)}"
        stats_text = f"Total positions: {total_positions}\nFrames: {frames_range}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Player heatmap saved to: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_team_comparison(self, data: Dict[str, Any], output_path: str = None, show: bool = True):
        """
        Plot side-by-side team heatmaps for comparison
        
        Args:
            data: Position data dictionary
            output_path: Path to save the plot
            show: Whether to display the plot
        """
        # Create team heatmaps
        team0_heatmap = self.create_team_heatmap(data, 0)
        team1_heatmap = self.create_team_heatmap(data, 1)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot Team 0
        im1 = ax1.imshow(team0_heatmap, cmap='Reds', interpolation='bilinear',
                        extent=[0, self.field_width, 0, self.field_height],
                        origin='lower', alpha=0.7)
        self._draw_field_lines(ax1)
        ax1.set_title('Team 0 Heatmap', fontsize=16, color=self.team_colors[0])
        ax1.set_xlabel('Field Width (pixels)', fontsize=12)
        ax1.set_ylabel('Field Height (pixels)', fontsize=12)
        
        # Plot Team 1
        im2 = ax2.imshow(team1_heatmap, cmap='Greens', interpolation='bilinear',
                        extent=[0, self.field_width, 0, self.field_height],
                        origin='lower', alpha=0.7)
        self._draw_field_lines(ax2)
        ax2.set_title('Team 1 Heatmap', fontsize=16, color=self.team_colors[1])
        ax2.set_xlabel('Field Width (pixels)', fontsize=12)
        ax2.set_ylabel('Field Height (pixels)', fontsize=12)
        
        # Add colorbars
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Position Density', fontsize=12)
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Position Density', fontsize=12)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Team comparison saved to: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_ball_heatmap(self, data: Dict[str, Any], output_path: str = None, show: bool = True):
        """
        Plot ball position heatmap
        
        Args:
            data: Position data dictionary
            output_path: Path to save the plot
            show: Whether to display the plot
        """
        ball_positions = data.get('ball', [])
        
        if not ball_positions:
            print("No ball position data found")
            return
        
        # Create heatmap
        heatmap = self.create_player_heatmap(ball_positions)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot heatmap
        im = ax.imshow(heatmap, cmap='YlOrRd', interpolation='bilinear',
                      extent=[0, self.field_width, 0, self.field_height],
                      origin='lower', alpha=0.7)
        
        # Add field lines
        self._draw_field_lines(ax)
        
        # Customize plot
        ax.set_title('Ball Position Heatmap', fontsize=16, color=self.team_colors['ball'])
        ax.set_xlabel('Field Width (pixels)', fontsize=12)
        ax.set_ylabel('Field Height (pixels)', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Position Density', fontsize=12)
        
        # Add statistics
        total_positions = len(ball_positions)
        if ball_positions:
            frames_range = f"{min(pos[2] for pos in ball_positions)} - {max(pos[2] for pos in ball_positions)}"
            stats_text = f"Total positions: {total_positions}\nFrames: {frames_range}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Ball heatmap saved to: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_movement_trails(self, data: Dict[str, Any], jersey_number: str, 
                             output_path: str = None, show: bool = True, 
                             trail_length: int = 100):
        """
        Create movement trail visualization for a player
        
        Args:
            data: Position data dictionary
            jersey_number: Player's jersey number
            output_path: Path to save the plot
            show: Whether to display the plot
            trail_length: Number of recent positions to show in trail
        """
        if jersey_number not in data['players']:
            print(f"Player {jersey_number} not found in data")
            return
        
        player_data = data['players'][jersey_number]
        positions = player_data['positions']
        team_id = player_data['team']
        
        if not positions:
            print(f"No position data for player {jersey_number}")
            return
        
        # Sort positions by frame
        positions = sorted(positions, key=lambda x: x[2])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw field
        self._draw_field_lines(ax, fill=True)
        
        # Plot movement trail
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        # Create gradient effect for trail
        for i in range(1, len(positions)):
            alpha = min(1.0, i / trail_length) if trail_length > 0 else 1.0
            ax.plot([x_coords[i-1], x_coords[i]], [y_coords[i-1], y_coords[i]], 
                   color=self.team_colors[team_id], alpha=alpha, linewidth=2)
        
        # Mark start and end positions
        if positions:
            ax.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
            ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
        
        # Customize plot
        ax.set_title(f'Player {jersey_number} Movement Trail (Team {team_id})', fontsize=16)
        ax.set_xlabel('Field Width (pixels)', fontsize=12)
        ax.set_ylabel('Field Height (pixels)', fontsize=12)
        ax.legend()
        
        # Set axis limits
        ax.set_xlim(0, self.field_width)
        ax.set_ylim(0, self.field_height)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Movement trail saved to: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def _draw_field_lines(self, ax, fill: bool = False):
        """Draw football field lines on the plot"""
        # Field boundaries
        margin = 20
        field_left, field_right = margin, self.field_width - margin
        field_top, field_bottom = self.field_height - margin, margin
        
        if fill:
            # Fill field with green color
            ax.add_patch(plt.Rectangle((field_left, field_bottom), 
                                     field_right - field_left, 
                                     field_top - field_bottom, 
                                     facecolor='lightgreen', alpha=0.3))
        
        # Field boundary
        ax.plot([field_left, field_right, field_right, field_left, field_left],
               [field_bottom, field_bottom, field_top, field_top, field_bottom],
               'white', linewidth=2)
        
        # Center line
        center_x = (field_left + field_right) / 2
        ax.plot([center_x, center_x], [field_bottom, field_top], 'white', linewidth=2)
        
        # Center circle
        center_y = (field_top + field_bottom) / 2
        circle = plt.Circle((center_x, center_y), 
                           min(self.field_width, self.field_height) * 0.1, 
                           fill=False, color='white', linewidth=2)
        ax.add_patch(circle)
    
    def generate_all_heatmaps(self, data_file: str, output_dir: str = "heatmaps"):
        """
        Generate all types of heatmaps and save them
        
        Args:
            data_file: Path to position data JSON file
            output_dir: Directory to save all heatmaps
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        data = self.load_position_data(data_file)
        
        print("Generating heatmaps...")
        
        # Team comparison
        self.plot_team_comparison(
            data, 
            output_path=os.path.join(output_dir, "team_comparison.png"),
            show=False
        )
        
        # Ball heatmap
        self.plot_ball_heatmap(
            data,
            output_path=os.path.join(output_dir, "ball_heatmap.png"),
            show=False
        )
        
        # Individual player heatmaps
        for jersey_number in data['players']:
            player_output = os.path.join(output_dir, f"player_{jersey_number}_heatmap.png")
            self.plot_player_heatmap(
                data, jersey_number,
                output_path=player_output,
                show=False
            )
            
            # Movement trails
            trail_output = os.path.join(output_dir, f"player_{jersey_number}_trail.png")
            self.create_movement_trails(
                data, jersey_number,
                output_path=trail_output,
                show=False
            )
        
        print(f"All heatmaps generated and saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate heatmaps from football position data")
    parser.add_argument("--data_file", type=str, required=True, 
                       help="Path to position data JSON file")
    parser.add_argument("--output_dir", type=str, default="output/heatmaps",
                       help="Directory to save heatmaps")
    parser.add_argument("--player", type=str, default=None,
                       help="Generate heatmap for specific player (jersey number)")
    parser.add_argument("--type", type=str, choices=['all', 'team', 'ball', 'player', 'trail'],
                       default='all', help="Type of heatmap to generate")
    parser.add_argument("--field_width", type=int, default=400,
                       help="Field width in pixels")
    parser.add_argument("--field_height", type=int, default=300,
                       help="Field height in pixels")
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data_file):
        print(f"Error: Data file {args.data_file} not found")
        return
    
    # Initialize heatmap generator
    generator = HeatmapGenerator((args.field_width, args.field_height))
    
    # Load data
    data = generator.load_position_data(args.data_file)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate requested heatmaps
    if args.type == 'all':
        generator.generate_all_heatmaps(args.data_file, args.output_dir)
    elif args.type == 'team':
        generator.plot_team_comparison(
            data,
            output_path=os.path.join(args.output_dir, "team_comparison.png")
        )
    elif args.type == 'ball':
        generator.plot_ball_heatmap(
            data,
            output_path=os.path.join(args.output_dir, "ball_heatmap.png")
        )
    elif args.type == 'player':
        if args.player:
            generator.plot_player_heatmap(
                data, args.player,
                output_path=os.path.join(args.output_dir, f"player_{args.player}_heatmap.png")
            )
        else:
            print("Error: --player argument required for player heatmap")
    elif args.type == 'trail':
        if args.player:
            generator.create_movement_trails(
                data, args.player,
                output_path=os.path.join(args.output_dir, f"player_{args.player}_trail.png")
            )
        else:
            print("Error: --player argument required for movement trail")


if __name__ == "__main__":
    main()
