import os
from graphviz import Digraph

def generate_diagrams():
    # Define output directory
    output_dir = os.path.join('report', 'artifacts')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating diagrams in {output_dir}...")

    # ==========================================
    # Diagram 1: Gated-Attention CEM Architecture
    # ==========================================
    dot1 = Digraph('Gated_Attention_CEM', comment='Gated Attention CEM Architecture')
    dot1.attr(rankdir='TB', splines='ortho', nodesep='0.6', ranksep='0.6')
    dot1.attr('node', shape='box', style='rounded,filled', fillcolor='aliceblue', fontname='Helvetica', fontsize='12')
    dot1.attr('edge', fontname='Helvetica', fontsize='10')

    # Nodes
    dot1.node('z_tilde', '<<B>z&#771;<sub>c</sub></B><BR/>Class Features<BR/>[M, D]>', shape='ellipse', fillcolor='lightyellow')
    dot1.node('ln', 'LayerNorm')
    
    # Gated Attention Pooling Block
    with dot1.subgraph(name='cluster_gate_pooling') as c:
        c.attr(label='Gated Attention Pooling', style='dashed', color='gray50', fontcolor='gray50')
        c.attr('node', fillcolor='honeydew')
        c.node('V', 'tanh(W<sub>V</sub> z&#771;<sub>c</sub>)')
        c.node('U', '&#963;(W<sub>U</sub> z&#771;<sub>c</sub>)')
        c.node('hadamard', '&#8857;', shape='circle', width='0.4', fixedsize='true', fillcolor='gold')
        c.node('w', 'Linear (w)')
        c.node('softmax', 'Softmax')
        c.node('alpha', '<<B>&#945;<sub>b</sub></B><BR/>Attention Weights<BR/>[M, 1]>', shape='ellipse', fillcolor='lightyellow')
        
        c.edge('V', 'hadamard')
        c.edge('U', 'hadamard')
        c.edge('hadamard', 'w')
        c.edge('w', 'softmax')
        c.edge('softmax', 'alpha')

    # Statistics & Loss
    dot1.node('weighted_mean', 'Weighted Mean &#956;<sub>c</sub>')
    dot1.node('weighted_var', 'Weighted Variance v<sub>c</sub>')
    dot1.node('log_var', 'log(v<sub>c</sub> + &#949;)')
    dot1.node('hinge', 'ReLU(log &#964; - log v<sub>c</sub>)', fillcolor='mistyrose')
    dot1.node('L_C', '<<B>L<sub>C</sub><sup>gated</sup></B><BR/>Privacy Loss>', shape='doubleoctagon', fillcolor='lightcoral')

    # Edges
    dot1.edge('z_tilde', 'ln')
    dot1.edge('ln', 'V')
    dot1.edge('ln', 'U')
    
    dot1.edge('ln', 'weighted_mean', label=' features')
    dot1.edge('alpha', 'weighted_mean', label=' weights')
    
    dot1.edge('ln', 'weighted_var', label=' features')
    dot1.edge('alpha', 'weighted_var', label=' weights')
    dot1.edge('weighted_mean', 'weighted_var', style='dashed', label='center')
    
    dot1.edge('weighted_var', 'log_var')
    dot1.edge('log_var', 'hinge')
    dot1.edge('hinge', 'L_C')

    # Render
    output_path1 = os.path.join(output_dir, 'fig_arch_gated_att')
    dot1.render(output_path1, format='png', cleanup=True)
    print(f"Saved: {output_path1}.png")


    # ==========================================
    # Diagram 2: Slot + Gated Cross-Attention CEM
    # ==========================================
    dot2 = Digraph('Slot_Cross_Attention_CEM', comment='Slot + Gated Cross-Attention CEM Architecture')
    dot2.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.7')
    dot2.attr('node', shape='box', style='rounded,filled', fillcolor='aliceblue', fontname='Helvetica', fontsize='12')
    dot2.attr('edge', fontname='Helvetica', fontsize='10')

    # Inputs
    dot2.node('z_tilde', '<<B>z&#771;<sub>c</sub></B><BR/>Class Features<BR/>[M, D]>', shape='ellipse', fillcolor='lightyellow')
    dot2.node('init_slots', 'Init Slots<BR/>N(&#956;<sub>&#952;</sub>, &#963;<sub>&#952;</sub><sup>2</sup>)', shape='ellipse', fillcolor='lightgrey')

    # 1. Slot Inference
    with dot2.subgraph(name='cluster_slot_inference') as c:
        c.attr(label='1. Slot Inference (Learned Mixtures)', style='dashed', color='gray50', fontcolor='gray50')
        c.attr('node', fillcolor='lavender')
        c.node('slot_attn', 'Slot Attention Module\n(Iterative GRU Updates)')
        c.node('slots', '<<B>Slots S</B><BR/>[S, D]>', shape='ellipse', fillcolor='moccasin')
        c.edge('slot_attn', 'slots')

    # 2. Gated Cross-Attention
    with dot2.subgraph(name='cluster_cross_attn') as c:
        c.attr(label='2. Gated Cross-Attention Refinement', style='dashed', color='gray50', fontcolor='gray50')
        c.attr('node', fillcolor='honeydew')
        c.node('cross_attn', 'Multi-Head Cross-Attn')
        c.node('gate_xattn', 'Gate: tanh(&#945;<sub>xattn</sub>)', shape='note', fillcolor='lemonchiffon')
        c.node('add_norm', 'Add & Norm')
        c.node('ffn', 'FFN + Gate: tanh(&#945;<sub>ffn</sub>)')
        c.node('enhanced_T', '<<B>T&#771;<sub>c</sub></B><BR/>Refined Features<BR/>[M, D]>', shape='ellipse', fillcolor='thistle')
        
        c.edge('cross_attn', 'add_norm')
        c.edge('gate_xattn', 'add_norm', style='dotted')
        c.edge('add_norm', 'ffn')
        c.edge('ffn', 'enhanced_T')

    # 3. Dispersion Penalty
    with dot2.subgraph(name='cluster_penalty') as c:
        c.attr(label='3. Dispersion Penalty Calculation', style='dashed', color='gray50', fontcolor='gray50')
        c.attr('node', fillcolor='mistyrose')
        
        c.node('resp', 'Responsibilities r<sub>mk</sub>\n(Softmax Cosine)')
        c.node('moments', 'Slot Moments\n(&#956;<sub>k</sub>, v<sub>k</sub>)')
        
        c.node('gates', 'Gates:\n1. MLP(LN(log v))\n2. Sigmoid(SNR)', shape='note', fillcolor='lemonchiffon')
        c.node('softplus', 'Softplus Threshold\nlog v<sub>k</sub> < log &#964;')
        
        c.node('mass', 'Slot Mass m<sub>k</sub><sup>p</sup>\n(Sharpened)')
        c.node('agg', 'Weighted Aggregation')
        
        c.edge('resp', 'moments')
        c.edge('resp', 'mass')
        c.edge('moments', 'gates')
        c.edge('moments', 'softplus')
        c.edge('gates', 'agg')
        c.edge('softplus', 'agg')
        c.edge('mass', 'agg')

    dot2.node('L_C_slot', '<<B>L<sub>C</sub><sup>slot</sup></B><BR/>Privacy Loss>', shape='doubleoctagon', fillcolor='lightcoral')

    # Global Edges
    dot2.edge('z_tilde', 'slot_attn', label=' tokens')
    dot2.edge('init_slots', 'slot_attn')
    
    dot2.edge('z_tilde', 'cross_attn', label=' Query')
    dot2.edge('slots', 'cross_attn', label=' Key/Value')
    
    dot2.edge('enhanced_T', 'resp', label=' T&#771;<sub>c</sub>')
    dot2.edge('slots', 'resp', label=' S')
    
    dot2.edge('agg', 'L_C_slot')

    # Render
    output_path2 = os.path.join(output_dir, 'fig_arch_slot_att')
    dot2.render(output_path2, format='png', cleanup=True)
    print(f"Saved: {output_path2}.png")

if __name__ == "__main__":
    try:
        generate_diagrams()
        print("\nSuccess! Diagrams are in 'report/artifacts/'.")
    except ImportError:
        print("Error: 'graphviz' python library is missing. Please run: pip install graphviz")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Note: You also need 'graphviz' installed on your system (e.g., 'brew install graphviz' or 'apt-get install graphviz').")
