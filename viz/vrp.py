import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use Agg backend instead of TkAgg to avoid tkinter dependency
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os

def read_vrp_file(filename):
    nodes = {}
    demands = {}
    depot = None
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    reading_nodes = False
    reading_demands = False
    
    for line in lines:
        if 'NODE_COORD_SECTION' in line:
            reading_nodes = True
            continue
        elif 'DEMAND_SECTION' in line:
            reading_nodes = False
            reading_demands = True
            continue
        elif 'DEPOT_SECTION' in line:
            reading_demands = False
            continue
            
        if reading_nodes:
            parts = line.strip().split()
            if len(parts) == 3:
                node_id, x, y = map(float, parts)
                nodes[int(node_id)] = (x, y)
                
        if reading_demands:
            parts = line.strip().split()
            if len(parts) == 2:
                node_id, demand = map(int, parts)
                demands[node_id] = demand

    return nodes, demands

def visualize_vrp(nodes, demands, routes=None, save_path=None, show_plot=False):
    G = nx.Graph()
    
    # Add nodes
    for node_id, coords in nodes.items():
        G.add_node(node_id, pos=coords, demand=demands.get(node_id, 0))
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw nodes
    depot_nodes = [n for n, d in G.nodes(data=True) if demands.get(n, 0) == 0]
    customer_nodes = [n for n, d in G.nodes(data=True) if demands.get(n, 0) > 0]
    
    # Draw depot (larger red node)
    nx.draw_networkx_nodes(G, pos, nodelist=depot_nodes, 
                          node_color='red', node_size=200)
    
    # Draw customers (blue nodes)
    nx.draw_networkx_nodes(G, pos, nodelist=customer_nodes, 
                          node_color='blue', node_size=100)
    
    # Draw node labels
    labels = {n: f"{n}\n(d:{demands[n]})" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels)
    
    # Draw routes if provided
    if routes:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))
        
        # For legend
        legend_elements = []
        
        for i, (route, color) in enumerate(zip(routes, colors)):
            # Ensure route visualization is a closed loop if it doesn't already return to depot
            if route[0] != route[-1] and depot_nodes and depot_nodes[0] not in [route[0], route[-1]]:
                # Create a closed route that starts and ends at depot
                closed_route = [depot_nodes[0]] + route + [depot_nodes[0]] if depot_nodes else route
            else:
                closed_route = route
                
            # Draw the route with arrows to show direction
            path_edges = list(zip(closed_route[:-1], closed_route[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                                edge_color=[color], width=2, arrows=True, 
                                arrowstyle='-|>', arrowsize=15)
            
            # Add to legend
            from matplotlib.lines import Line2D
            legend_elements.append(Line2D([0], [0], color=color, lw=2, label=f'Route #{i+1}'))
            
            # Add route annotations - mark order of visit
            edge_labels = {(closed_route[j], closed_route[j+1]): f"{j+1}" 
                         for j in range(len(closed_route)-1)}
            
            # Optional - can uncomment to show visit order on edges
            # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color=color)
        
        # Add the legend
        plt.legend(handles=legend_elements, loc='best')
    
    plt.title("VRP Instance Visualization")
    plt.axis('equal')
    plt.grid(True)
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # Show the plot if requested
    if show_plot:
        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display plot: {e}")
            print("Plot was saved to file instead.")
    
    plt.close()  # Close figure to free up memory

def visualize_vrp_interactive(nodes, demands, routes=None, save_path='vrp_interactive.html'):
    """
    Create an interactive HTML visualization of the VRP problem using Plotly.
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        print("Plotly is required for interactive visualization.")
        print("Please install it using: pip install plotly")
        return None
    
    try:
        # Create figure
        fig = go.Figure()
        
        # Identify depot and customer nodes
        depot_nodes = [n for n, d in demands.items() if d == 0]
        depot_id = depot_nodes[0] if depot_nodes else 1
        customer_nodes = [n for n, d in demands.items() if d > 0]
        
        # Add depot nodes
        depot_x = [nodes[n][0] for n in depot_nodes]
        depot_y = [nodes[n][1] for n in depot_nodes]
        fig.add_trace(go.Scatter(
            x=depot_x, y=depot_y,
            mode='markers+text',
            marker=dict(
                symbol='square',
                size=20,
                color='red',
                line=dict(width=2, color='darkred')
            ),
            text=[f"{n}" for n in depot_nodes],
            textposition="top center",
            name='Depot',
            hoverinfo='text',
            hovertext=[f"Depot {n}<br>Demand: {demands[n]}<br>Coords: ({nodes[n][0]:.2f}, {nodes[n][1]:.2f})" for n in depot_nodes]
        ))
        
        # Add customer nodes
        customer_x = [nodes[n][0] for n in customer_nodes]
        customer_y = [nodes[n][1] for n in customer_nodes]
        fig.add_trace(go.Scatter(
            x=customer_x, y=customer_y,
            mode='markers+text',
            marker=dict(
                symbol='circle',
                size=15,
                color='blue',
                line=dict(width=1, color='darkblue')
            ),
            text=[f"{n}<br>d:{demands[n]}" for n in customer_nodes],
            textposition="top center",
            name='Customers',
            hoverinfo='text',
            hovertext=[f"Customer {n}<br>Demand: {demands[n]}<br>Coords: ({nodes[n][0]:.2f}, {nodes[n][1]:.2f})" for n in customer_nodes]
        ))
        
        # Add routes if provided
        if routes:
            colors = px.colors.qualitative.Plotly if hasattr(px.colors.qualitative, 'Plotly') else \
                     ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            for i, route in enumerate(routes):
                color = colors[i % len(colors)]
                
                # Create a closed route if needed
                if route[0] != route[-1] and depot_id not in [route[0], route[-1]]:
                    closed_route = [depot_id] + route + [depot_id]
                else:
                    closed_route = route
                    
                # Extract x, y coordinates for the route
                route_x = [nodes[n][0] for n in closed_route]
                route_y = [nodes[n][1] for n in closed_route]
                
                # Add a line trace for the route
                fig.add_trace(go.Scatter(
                    x=route_x, y=route_y,
                    mode='lines+markers',
                    line=dict(width=2, color=color),
                    marker=dict(size=8),
                    name=f'Route #{i+1}',
                    hoverinfo='text',
                    hovertext=[f"Visit {j+1} - Node {n}" for j, n in enumerate(closed_route)]
                ))
                
                # Add arrows to show direction
                for j in range(len(closed_route)-1):
                    x0, y0 = nodes[closed_route[j]]
                    x1, y1 = nodes[closed_route[j+1]]
                    mid_x = x0 + (x1 - x0) * 0.6  # Arrow position at 60% of the way
                    mid_y = y0 + (y1 - y0) * 0.6
                    
                    fig.add_annotation(
                        x=mid_x, y=mid_y,
                        ax=x0, ay=y0,
                        xref="x", yref="y",
                        axref="x", ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.5,
                        arrowwidth=2,
                        arrowcolor=color
                    )
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Save the figure to HTML with full plotly.js
        fig.write_html(
            save_path,
            include_plotlyjs=True,  # Changed from 'cdn' to True
            full_html=True
        )
        print(f"Interactive visualization saved to {save_path}")
        return fig
        
    except Exception as e:
        print(f"Error creating interactive visualization: {e}")
        return None

def parse_solution(solution_file, depot_id=1):
    routes = []
    with open(solution_file, 'r') as f:
        for line in f:
            if line.startswith('Route'):
                # Extract just the customer nodes from the solution
                customer_nodes = [int(x) for x in line.split(':')[1].strip().split()]
                
                # Create full route starting and ending at depot
                if depot_id not in customer_nodes:
                    route = [depot_id] + customer_nodes + [depot_id]
                elif customer_nodes[0] != depot_id or customer_nodes[-1] != depot_id:
                    # If depot is in the customer list but not as start and end, rebuild properly
                    if depot_id in customer_nodes:
                        # Remove depot from customer list to avoid duplicates
                        customer_nodes = [node for node in customer_nodes if node != depot_id]
                    route = [depot_id] + customer_nodes + [depot_id]
                else:
                    route = customer_nodes
                    
                routes.append(route)
    return routes

# Example usage
def run_visualization(vrp_file='toy.vrp', solution_file='sol.vrp', 
                     output_file='vrp_solution.png', html_output='vrp_interactive.html'):
    try:
        nodes, demands = read_vrp_file(vrp_file)
        routes = parse_solution(solution_file)
        
        # Print route information
        for i, route in enumerate(routes):
            total_demand = sum(demands.get(node, 0) for node in route)
            print(f"Route #{i+1}: {' -> '.join(map(str, route))}, Total Demand: {total_demand}")
        
        # Set default output directory to current directory
        output_path = os.path.join(os.getcwd(), output_file)
        html_path = os.path.join(os.getcwd(), html_output)
        
        # Visualize and save to static file
        visualize_vrp(nodes, demands, routes, save_path=output_path)
        
        # Create interactive HTML visualization
        fig = visualize_vrp_interactive(nodes, demands, routes, save_path=html_path)
        if fig is not None:
            print(f"Interactive HTML visualization saved to: {html_path}")
        else:
            print("Failed to create interactive visualization")
        
        print(f"Static visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"Error in visualization: {e}")

# For parMDS visualization
def visualize_parMDS(parMDS_results, save_path=None, show_plot=False):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Assuming parMDS_results is a numpy array with shape (n_points, 3)
    scatter = ax.scatter(parMDS_results[:, 0], 
                        parMDS_results[:, 1], 
                        parMDS_results[:, 2],
                        c=range(len(parMDS_results)),
                        cmap='viridis')
    
    plt.colorbar(scatter)
    ax.set_title('parMDS Visualization')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"parMDS plot saved to {save_path}")
    
    if show_plot:
        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display parMDS plot: {e}")
    
    plt.close()

if __name__ == "__main__":
    run_visualization()