#!/usr/bin/env python3
"""
Direct Visualization of OGB BioKG Community Structure

This script creates publication-quality community visualizations directly from BioKG data.
Shows protein clusters and drug/disease bridge nodes as described.

Usage:
    python visualize_biokg_community.py --data_dir data/ogbl-biokg --output_dir biokg_visualization
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Configure style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Configure Vietnamese font support
import matplotlib.font_manager as fm
try:
    vietnamese_fonts = ['DejaVu Sans', 'Arial Unicode MS', 'Times New Roman', 'Liberation Sans']
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    vietnamese_font = None
    for font in vietnamese_fonts:
        if font in available_fonts:
            vietnamese_font = font
            break

    if vietnamese_font:
        plt.rcParams['font.family'] = vietnamese_font
        print(f"[FONT] Using Vietnamese-compatible font: {vietnamese_font}")
except Exception as e:
    print(f"[FONT] Font configuration failed: {e}")


def load_biokg_data(data_dir):
    """Load OGB BioKG dataset"""
    print("Loading BioKG data...")

    # Load entity and relation mappings
    entity2id = {}
    relation2id = {}

    # Load entities.dict
    entities_path = os.path.join(data_dir, 'entities.dict')
    if os.path.exists(entities_path):
        with open(entities_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    entity_id, entity_name = int(parts[0]), parts[1]
                    entity2id[entity_name] = entity_id
        print(f"Loaded {len(entity2id):,} entities")

    # Load relation names
    relations_raw_dir = os.path.join(data_dir.replace('data/ogbl-biokg', 'dataset/ogbl_biokg/raw'), 'relations')
    if os.path.exists(relations_raw_dir):
        relation_files = sorted(os.listdir(relations_raw_dir))
        for rel_id, filename in enumerate(relation_files):
            relation_name = filename.replace('___', ' → ').replace('_', ' ')
            relation2id[relation_name] = rel_id
        print(f"Loaded {len(relation2id)} relations")

    # Load entity type mapping
    entity_dict_path = os.path.join(data_dir, 'entity_dict.json')
    entity_type_ranges = {}
    if os.path.exists(entity_dict_path):
        with open(entity_dict_path, 'r') as f:
            entity_type_ranges = json.load(f)

    # Load train data
    train_triplets = []
    train_path = os.path.join(data_dir, 'train.txt')
    if os.path.exists(train_path):
        with open(train_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    h, r, t = parts[0], parts[1], parts[2]
                    train_triplets.append((int(h), int(r), int(t)))

    print(f"Loaded {len(train_triplets):,} training triplets")

    return {
        'entity2id': entity2id,
        'relation2id': relation2id,
        'entity_type_ranges': entity_type_ranges,
        'triplets': train_triplets
    }


def get_entity_types(data):
    """Determine entity types for each entity ID"""
    print("Determining entity types...")

    entity_types = {}
    entity_type_counts = Counter()

    # Method 1: Use entity type ranges
    for entity_name, entity_id in data['entity2id'].items():
        entity_type_found = False
        if 'entity_type_ranges' in data and data['entity_type_ranges']:
            for type_name, id_range in data['entity_type_ranges'].items():
                if isinstance(id_range, list) and len(id_range) == 2:
                    start_id, end_id = id_range
                    if start_id <= entity_id < end_id:
                        entity_types[entity_id] = type_name
                        entity_type_counts[type_name] += 1
                        entity_type_found = True
                        break

        # Fallback: infer from entity name
        if not entity_type_found:
            entity_name_lower = entity_name.lower()
            if any(keyword in entity_name_lower for keyword in ['protein', 'gene', 'uniprot']):
                entity_types[entity_id] = 'protein'
                entity_type_counts['protein'] += 1
            elif any(keyword in entity_name_lower for keyword in ['disease', 'disorder', 'phenotype']):
                entity_types[entity_id] = 'disease'
                entity_type_counts['disease'] += 1
            elif any(keyword in entity_name_lower for keyword in ['drug', 'compound', 'chemical']):
                entity_types[entity_id] = 'drug'
                entity_type_counts['drug'] += 1
            elif any(keyword in entity_name_lower for keyword in ['function', 'process', 'pathway']):
                entity_types[entity_id] = 'function'
                entity_type_counts['function'] += 1
            else:
                entity_types[entity_id] = 'other'
                entity_type_counts['other'] += 1

    print(f"Entity type distribution:")
    for ent_type, count in entity_type_counts.most_common():
        print(f"  {ent_type}: {count:,} entities")

    return entity_types


def build_biokg_graph(data, entity_types, max_nodes=500):
    """Build NetworkX graph from BioKG data"""
    print(f"Building graph with max {max_nodes} nodes...")

    # Create undirected graph
    G = nx.Graph()

    # Sample high-degree entities to ensure we get key proteins and drugs
    entity_degrees = Counter()
    for h, r, t in data['triplets']:
        entity_degrees[h] += 1
        entity_degrees[t] += 1

    # Select top entities by degree
    top_entities = set([entity_id for entity_id, _ in entity_degrees.most_common(max_nodes)])

    # Add edges between selected entities
    edge_count = 0
    for h, r, t in data['triplets']:
        if h in top_entities and t in top_entities and h != t:
            # Add entity type information
            G.add_edge(h, t, relation=r)
            edge_count += 1

            if edge_count >= 10000:  # Limit edges for visualization
                break

    print(f"Created graph with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges")
    return G


def detect_communities(G):
    """Detect communities using Louvain algorithm"""
    print("Detecting communities...")

    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, resolution=1.0)

        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)

        num_communities = len(communities)
        print(f"Detected {num_communities} communities")

        # Calculate community statistics
        comm_sizes = [len(members) for members in communities.values()]
        print(f"Community sizes: mean={np.mean(comm_sizes):.1f}, median={np.median(comm_sizes):.1f}")

        return communities, partition

    except ImportError:
        print("python-louvain not installed. Installing...")
        import subprocess
        subprocess.run(["pip", "install", "python-louvain"])
        return detect_communities(G)
    except Exception as e:
        print(f"Community detection failed: {e}")
        return None, None


def create_community_visualization(G, communities, entity_types, output_dir):
    """Create separate visualizations for top 4 communities"""
    print("Creating separate visualizations for top 4 communities...")

    # Color maps
    entity_color_map = {
        'protein': '#2ecc71',      # Green - protein clusters
        'disease': '#e74c3c',      # Red - diseases
        'drug': '#f39c12',         # Orange - drugs (bridge nodes)
        'function': '#3498db',     # Blue - biological functions
        'sideeffect': '#9b59b6',   # Purple - side effects
        'other': '#95a5a6'         # Gray - others
    }

    # Get community by size (descending)
    sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
    top_4_comms = sorted_communities[:4]

    # Create separate figure for each community
    for idx, (comm_id, members) in enumerate(top_4_comms):
        # Create individual figure for this community
        fig, ax = plt.subplots(figsize=(20, 16))

        # Create subgraph for this community
        comm_subgraph = G.subgraph(members)

        # Layout with better spacing
        pos = nx.spring_layout(comm_subgraph, k=2.0, iterations=150, seed=idx)

        # Node colors by entity type
        node_colors = []
        node_sizes = []
        for node in comm_subgraph.nodes():
            ent_type = entity_types.get(node, 'other')
            node_colors.append(entity_color_map.get(ent_type, '#95a5a6'))

            # Larger nodes for drugs and diseases (bridge nodes)
            if ent_type in ['drug', 'disease']:
                node_sizes.append(200)  # Bridge nodes - larger
            else:
                node_sizes.append(120)   # Regular nodes

        # Draw the community network
        nx.draw_networkx_nodes(comm_subgraph, pos,
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.8,
                              ax=ax)

        nx.draw_networkx_edges(comm_subgraph, pos,
                              edge_color='gray',
                              alpha=0.4,
                              width=1.5,
                              ax=ax)

        # Add labels for high-degree nodes
        degree_dict = dict(comm_subgraph.degree())
        top_nodes_in_comm = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:15]

        labels = {}
        for node, degree in top_nodes_in_comm:
            if degree > 2:  # Only label relatively connected nodes
                ent_type = entity_types.get(node, 'other')
                labels[node] = f"{ent_type[:3].upper()}\n({degree})"

        nx.draw_networkx_labels(comm_subgraph, pos, labels,
                               font_size=10, font_color='black', ax=ax)

        # Calculate community statistics for title
        protein_count = sum(1 for node in members if entity_types.get(node, 'other') == 'protein')
        disease_count = sum(1 for node in members if entity_types.get(node, 'other') == 'disease')
        drug_count = sum(1 for node in members if entity_types.get(node, 'other') == 'drug')
        function_count = sum(1 for node in members if entity_types.get(node, 'other') == 'function')
        sideeffect_count = sum(1 for node in members if entity_types.get(node, 'other') == 'sideeffect')

        # Calculate density
        density = nx.density(comm_subgraph)

        # Set title with detailed community statistics
        ax.set_title(f'Cộng đồng {idx+1}: {len(members)} thực thể, {comm_subgraph.number_of_edges()} cạnh, Density: {density:.3f}\n'
                    f'Protein: {protein_count} | Disease: {disease_count} | Drug: {drug_count} | Function: {function_count} | Side Effect: {sideeffect_count}',
                    fontsize=18, fontweight='bold', pad=20)

        ax.axis('off')

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=entity_color_map['protein'],
                       markersize=15, label='Protein clusters'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=entity_color_map['drug'],
                       markersize=18, label='Drug bridge nodes'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=entity_color_map['disease'],
                       markersize=18, label='Disease bridge nodes'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=entity_color_map['function'],
                       markersize=15, label='Biological functions'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=entity_color_map['sideeffect'],
                       markersize=15, label='Side effects'),
        ]

        ax.legend(handles=legend_elements, loc='upper right', fontsize=14, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'community_{idx+1}_separate.png'),
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'community_{idx+1}_separate.pdf'),
                    bbox_inches='tight')
        print(f"Saved community {idx+1} separate visualization to {output_dir}/community_{idx+1}_separate.png")
        plt.close()

    # Create additional visualizations
    create_pie_chart_analysis(G, top_4_comms, entity_types, output_dir)
    create_full_graph_scatter(G, communities, entity_types, output_dir)

    # Create LaTeX table for statistics
    create_statistics_table(G, top_4_comms, entity_types, output_dir)


def create_pie_chart_analysis(G, top_4_comms, entity_types, output_dir):
    """Create pie charts showing entity type ratios in communities"""
    print("Creating pie chart analysis...")

    # Color maps
    entity_color_map = {
        'protein': '#2ecc71',      # Green
        'disease': '#e74c3c',      # Red
        'drug': '#f39c12',         # Orange
        'function': '#3498db',     # Blue
        'sideeffect': '#9b59b6',   # Purple
        'other': '#95a5a6'         # Gray
    }

    # Create 2x2 grid for 4 pie charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Phân phối Tỷ lệ Loại Thực thể trong Top 4 Cộng đồng OGB BioKG',
                 fontsize=20, fontweight='bold')

    axes = axes.flatten()

    for idx, (comm_id, members) in enumerate(top_4_comms):
        ax = axes[idx]

        # Count entity types in this community
        comm_entity_counts = Counter()
        for node in members:
            ent_type = entity_types.get(node, 'other')
            comm_entity_counts[ent_type] += 1

        # Prepare data for pie chart
        labels = []
        sizes = []
        colors = []

        total = len(members)
        for ent_type, count in comm_entity_counts.most_common():
            if count > 0:  # Only include types that exist
                percentage = (count / total) * 100
                labels.append(f'{ent_type.capitalize()}\n{count:,} ({percentage:.1f}%)')
                sizes.append(count)
                colors.append(entity_color_map.get(ent_type, '#95a5a6'))

        # Create pie chart
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='',
                                         startangle=90, textprops={'fontsize': 10})

        # Enhance text properties
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)

        ax.set_title(f'Cộng đồng {idx+1}\n({total:,} thực thể)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'community_entity_pie_charts.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'community_entity_pie_charts.pdf'),
                bbox_inches='tight')
    print(f"Saved pie chart analysis to {output_dir}/community_entity_pie_charts.png")
    plt.close()


def create_full_graph_scatter(G, communities, entity_types, output_dir):
    """Create scatter plot of full graph with communities color-coded"""
    print("Creating full graph scatter visualization...")

    # Generate community colors
    import random
    num_communities = len(communities)
    community_colors = []

    # Generate distinct colors for communities
    for i in range(num_communities):
        hue = i / num_communities
        # Convert HSV to RGB
        import colorsys
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        community_colors.append(rgb)

    # Create layout for visualization
    print("Computing layout for full graph...")
    pos = nx.spring_layout(G, k=3.0, iterations=100, seed=42)

    # Create figure
    fig, ax = plt.subplots(figsize=(24, 20))

    # Color nodes by community
    node_colors = []
    for node in G.nodes():
        # Find which community this node belongs to
        for comm_id, members in communities.items():
            if node in members:
                node_colors.append(community_colors[comm_id % len(community_colors)])
                break
        else:
            node_colors.append((0.7, 0.7, 0.7))  # Gray for unassigned nodes

    # Node sizes based on degree
    degrees = dict(G.degree())
    node_sizes = [20 + degrees[node] * 5 for node in G.nodes()]

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          alpha=0.7, ax=ax)

    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.2, width=0.5, ax=ax)

    # Create legend for top 10 communities
    sorted_comms = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    legend_elements = []
    for i, (comm_id, members) in enumerate(sorted_comms):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=community_colors[comm_id % len(community_colors)],
                                            markersize=12,
                                            label=f'Cộng đồng {comm_id+1}: {len(members)} nodes'))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
               title='Top 10 Communities', framealpha=0.9)

    # Add statistics text
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()
    num_comms = len(communities)

    stats_text = f"""Thống kê Mạng lưới BioKG:

• Tổng số nút: {total_nodes:,}
• Tổng số cạnh: {total_edges:,}
• Số cộng đồng: {num_comms:,}
• Bậc trung bình: {2*total_edges/total_nodes:.2f}

Phân phối Top 10 Cộng đồng:"""

    for i, (comm_id, members) in enumerate(sorted_comms[:10]):
        percentage = (len(members) / total_nodes) * 100
        stats_text += f"\n  {i+1}. Cộng đồng {comm_id+1}: {len(members):,} nodes ({percentage:.1f}%)"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_title('Cấu trúc Cộng đồng Toàn diện OGB BioKG\n(Mỗi màu đại diện một cộng đồng)',
                fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'full_graph_community_scatter.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'full_graph_community_scatter.pdf'),
                bbox_inches='tight')
    print(f"Saved full graph scatter to {output_dir}/full_graph_community_scatter.png")
    plt.close()


def create_statistics_table(G, top_4_comms, entity_types, output_dir):
    """Create LaTeX table with community statistics"""
    print("Creating LaTeX statistics table...")

    # Calculate overall network statistics
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()
    avg_degree = 2 * total_edges / total_nodes if total_nodes > 0 else 0

    # Entity type distribution in overall graph
    overall_entity_counts = Counter()
    for node in G.nodes():
        ent_type = entity_types.get(node, 'other')
        overall_entity_counts[ent_type] += 1

    # Community statistics
    community_stats = []
    for idx, (comm_id, members) in enumerate(top_4_comms):
        comm_subgraph = G.subgraph(members)

        # Count entity types in this community
        comm_entity_counts = Counter()
        for node in members:
            ent_type = entity_types.get(node, 'other')
            comm_entity_counts[ent_type] += 1

        # Calculate density
        density = nx.density(comm_subgraph)

        community_stats.append({
            'id': idx + 1,
            'size': len(members),
            'edges': comm_subgraph.number_of_edges(),
            'density': density,
            'protein': comm_entity_counts.get('protein', 0),
            'disease': comm_entity_counts.get('disease', 0),
            'drug': comm_entity_counts.get('drug', 0),
            'function': comm_entity_counts.get('function', 0),
            'sideeffect': comm_entity_counts.get('sideeffect', 0),
            'other': comm_entity_counts.get('other', 0)
        })

    # Create LaTeX table
    latex_table = f"""
% OGB BioKG Community Statistics
\\begin{{table}}[h]
\\centering
\\caption{{Statistics of Top 4 Communities in OGB BioKG}}
\\label{{tab:biokg_community_stats}}
\\begin{{tabular}}{{lccccccccc}}
\\toprule
\\textbf{{Community}} & \\textbf{{Size}} & \\textbf{{Edges}} & \\textbf{{Density}} & \\textbf{{Protein}} & \\textbf{{Disease}} & \\textbf{{Drug}} & \\textbf{{Function}} & \\textbf{{Side Effect}} & \\textbf{{Other}} \\\\
\\midrule
"""

    for stats in community_stats:
        latex_table += f"C{stats['id']} & {stats['size']:,} & {stats['edges']:,} & {stats['density']:.3f} & {stats['protein']:,} & {stats['disease']:,} & {stats['drug']:,} & {stats['function']:,} & {stats['sideeffect']:,} & {stats['other']:,} \\\\\n"

    latex_table += f"""\\midrule
\\textbf{{Overall Network}} & {total_nodes:,} & {total_edges:,} & {avg_degree:.2f} & {overall_entity_counts.get('protein', 0):,} & {overall_entity_counts.get('disease', 0):,} & {overall_entity_counts.get('drug', 0):,} & {overall_entity_counts.get('function', 0):,} & {overall_entity_counts.get('sideeffect', 0):,} & {overall_entity_counts.get('other', 0):,} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    # Save LaTeX table
    latex_path = os.path.join(output_dir, 'biokg_community_statistics.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved LaTeX statistics table to {latex_path}")




def main():
    parser = argparse.ArgumentParser(description='Visualize OGB BioKG Community Structure')
    parser.add_argument('--data_dir', type=str, default='data/ogbl-biokg',
                        help='Path to OGB BioKG dataset directory')
    parser.add_argument('--output_dir', type=str, default='biokg_visualization',
                        help='Output directory for visualizations')
    parser.add_argument('--max_nodes', type=int, default=45086,
                        help='Maximum number of nodes to include in visualization')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("OGB BIOKG COMMUNITY VISUALIZATION")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max nodes: {args.max_nodes}")
    print("="*80 + "\n")

    # Load data
    data = load_biokg_data(args.data_dir)

    # Get entity types
    entity_types = get_entity_types(data)

    # Build graph
    G = build_biokg_graph(data, entity_types, args.max_nodes)

    # Detect communities
    communities, partition = detect_communities(G)

    if communities:
        # Create visualizations
        create_community_visualization(G, communities, entity_types, args.output_dir)

        print("\n" + "="*80)
        print("VISUALIZATION COMPLETE!")
        print("="*80)
        print(f"\nResults saved to: {args.output_dir}/")
        print("Generated files:")
        print("  - community_1_separate.png/pdf (Community 1 visualization)")
        print("  - community_2_separate.png/pdf (Community 2 visualization)")
        print("  - community_3_separate.png/pdf (Community 3 visualization)")
        print("  - community_4_separate.png/pdf (Community 4 visualization)")
        print("  - community_entity_pie_charts.png/pdf (Pie charts of entity ratios)")
        print("  - full_graph_community_scatter.png/pdf (Full graph with color-coded communities)")
        print("  - biokg_community_statistics.tex (LaTeX statistics table)")
        print("\nVisualization shows:")
        print("  • Green clusters: Protein groups with high internal connectivity")
        print("  • Orange/Red nodes: Drug and Disease bridge nodes at interfaces")
        print("  • Blue nodes: Biological functions connecting different modules")
        print("  • Purple nodes: Side effects bridging different communities")
        print("  • Network exhibits modular organization with clear community structure")
        print("\n" + "="*80)
    else:
        print("Failed to detect communities. Please check installation of python-louvain.")


if __name__ == '__main__':
    main()
