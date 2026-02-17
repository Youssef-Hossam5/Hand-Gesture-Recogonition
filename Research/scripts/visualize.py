import plotly.graph_objects as go

connections = [
    (1,2),(2,3),(3,4),(4,5),
    (1,6),(6,7),(7,8),(8,9),
    (1,10),(10,11),(11,12),(12,13),
    (1,14),(14,15),(15,16),(16,17),
    (1,18),(18,19),(19,20),(20,21),
    (6,10),(10,14),(14,18)
]

def prepare_coords(sample, flip_x=True, flip_y=True):
    xs = [sample[f'x{i}'] for i in range(1, 22)]
    ys = [sample[f'y{i}'] for i in range(1, 22)]
    if flip_x:
        xs = [max(xs) - x for x in xs]
    if flip_y:
        ys = [max(ys) - y for y in ys]
    return xs, ys

def plot_sample(sample, label):
    xs, ys = prepare_coords(sample)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', marker=dict(size=8, color='red')))
    for start, end in connections:
        fig.add_trace(go.Scatter(x=[xs[start-1], xs[end-1]], y=[ys[start-1], ys[end-1]],
                                 mode='lines', line=dict(color='green', width=3), showlegend=False))
    fig.update_layout(title=f"2D Hand Skeleton - Class {label}",
                      xaxis=dict(scaleanchor="y"), yaxis=dict(scaleanchor="x"),
                      width=600, height=600)
    fig.show()
