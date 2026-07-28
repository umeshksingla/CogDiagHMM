def get_plot_config(task_name):
    custom_pos, props = None, None
    size = 4.2
    if task_name == 'nbacktask':
        custom_pos = {
            0: ([0, 0]),
            1: ([1, 0]),
            2: ([2, 0]),
            3: ([3, 0]),
            4: ([4, 0]),
            5: ([5, 0]),
            6: ([6, 0]),
            7: ([7, 0]),
        }
        props = {
            'edge_rad': 0.3,
            'linear_rad': 0.3,
            'ymargin': 1,
        }
        size = (10, 10)
    elif task_name == 'orderedtask':
        custom_pos = {
            0: ([0, 0]),
            1: ([1, 0]),
            2: ([2, 0]),
            3: ([3, 0]),
            4: ([4, 0]),
        }
        props = {
            'edge_rad': 0.3,
            'ymargin': 2.0,
        }
        size = (7, 7)
    elif task_name == 'hierarchicalcuetask':
        custom_pos = {
            0: ([0, 0]),
            1: ([-1, -1]),
            2: ([1, -1]),
            3: ([-1.5, -2]),
            4: ([-0.5, -2]),
            5: ([0.5, -2]),
            6: ([1.5, -2]),
        }
        props = {
            'edge_rad': 0.1,
        }
        size = (5, 5)
    elif task_name == 'cyclicfwdtask':
        props = {
            'edge_rad': 0.3,
            'linear_rad': 0.3,
        }
    elif task_name == 'countingfinitetask':
        custom_pos = {
            0: ([0, 0]),
            1: ([1, 0]),
            2: ([2, 0]),
            3: ([3, 0]),
            4: ([4, 0]),
            5: ([5, 0]),
        }
        props = {
            'linear_rad': 0.0,
        }
        size = (7, 7)
    else:
        raise ValueError(f'Unknown task name: {task_name}')
    return custom_pos, props, size
