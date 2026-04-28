# ===========================================
# visualize.py — Interactive Plotly visualization
# ===========================================

import plotly.graph_objects as go


def plot_clusters(clean_embeddings_2d, cluster_labels, cluster_names, clean_texts, sentiment_labels, num_clusters=5):
    """
    Create an interactive scatter plot with hover text.
    Hover shows review text and sentiment.
    """
    fig = go.Figure()

    for cluster_num in range(num_clusters):

        cluster_x = []
        cluster_y = []
        cluster_text = []

        for i in range(len(cluster_labels)):
            if cluster_labels[i] == cluster_num:
                cluster_x.append(clean_embeddings_2d[i, 0])
                cluster_y.append(clean_embeddings_2d[i, 1])
                short_text = clean_texts[i][:100]
                hover = short_text + " | " + sentiment_labels[i]
                cluster_text.append(hover)

        fig.add_trace(go.Scatter(
            x=cluster_x,
            y=cluster_y,
            mode="markers",
            marker=dict(size=3, opacity=0.5),
            name=cluster_names[cluster_num],
            text=cluster_text,
            hoverinfo="text+name"
        ))

    fig.update_layout(
        width=800,
        height=600,
        legend=dict(font=dict(size=12))
    )

    return fig