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
    elif task_name == 'communitytask':
        custom_pos = {
            # Community A: upper left
            0: (-3.6, 2.2),  # A0, core
            1: (-3.8, 1.2),  # A1, core
            2: (-2.7, 2.8),  # A2, core
            3: (-1.8, 1.8),  # A3, boundary toward B3
            4: (-2.4, 0.8),  # A4, boundary toward C4

            # Community B: upper right
            5: (3.6, 2.2),  # B0, core
            6: (3.8, 1.2),  # B1, core
            7: (2.7, 2.6),  # B2, core
            8: (1.8, 1.8),  # B3, boundary toward A3
            9: (2.4, 0.8),  # B4, boundary toward C3

            # Community C: bottom
            10: (0.0, -3.4),  # C0, core
            11: (-0.9, -2.7),  # C1, core
            12: (0.9, -2.7),  # C2, core
            13: (0.7, -1.3),  # C3, boundary toward B4
            14: (-0.7, -1.3),  # C4, boundary toward A4
        }
        props = {
            'edge_rad': 0.0,
            'linear_rad': 0.0,
            'radius': 0.2,
        }
        size = (8, 8)
    else:
        raise ValueError(f'Unknown task name: {task_name}')
    return custom_pos, props, size
