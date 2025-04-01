import tkinter as tk
from tkinter import filedialog, Menu
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from tkinter import filedialog, Menu, messagebox
import json  # 新增json模块
from utils import *
from tkinter import filedialog, Menu, messagebox, simpledialog
import json
import os
import matplotlib.pyplot as plt
from tkinter import colorchooser
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
import traceback


class DataVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detect head motion events")

        # 初始化数据存储
        self.data_types = ['eye_angle', 'tail_angle', 'swimbladder_top',
                           'swimbladder_side', 'head_motion', 'jaw_motion']  
        self.manual_dtypes = ['tail_angle', 'head_motion', 'jaw_motion']  
        self.data = {
            'eye_angle': {'left': None, 'right': None},
            'tail_angle': None,
            'swimbladder_top': None,
            'swimbladder_side': None,
            'head_motion': None,
            'jaw_motion': None  
        }
        self.segments = {t: [] for t in self.data_types}
        self.thresholds = {t: None for t in self.data_types}
        self.selected_segment = None
        self.initial_thresholds = {}
        self.segment_annotation = None

        # 创建Matplotlib图形
        self.fig = Figure(figsize=(10, 8))  # 增大图形高度
        self.fig.subplots_adjust(
            left=0.1,
            right=0.95,
            bottom=0.03,  # 减小底部边距
            top=0.97,  # 减小顶部边距
            hspace=0.5  # 减小垂直间距
        )
        self.axes = {}
        self.threshold_lines = {}
        self.data_lines = {}

        for i, dtype in enumerate(self.data_types, 1):
            ax = self.fig.add_subplot(6, 1, i)  # 改为6行1列
            ax.set_title(dtype.replace('_', ' ').title())
            self.axes[dtype] = ax

            if dtype == 'eye_angle':
                self.data_lines[dtype] = {
                    'left': Line2D([], [], color='red', lw=1),
                    'right': Line2D([], [], color='blue', lw=1)
                }
                ax.add_line(self.data_lines[dtype]['left'])
                ax.add_line(self.data_lines[dtype]['right'])
            else:
                line = Line2D([], [], color='steelblue', lw=1)
                ax.add_line(line)
                self.data_lines[dtype] = line

            if dtype in self.manual_dtypes:
                threshold_line = Line2D([0, 1], [0, 0], color='green',
                                        linewidth=2, visible=False)
                ax.add_line(threshold_line)
                self.threshold_lines[dtype] = threshold_line

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.undo_stack = []
        self.root.bind('<Control-z>', self.undo_last_delete)

        # 初始化进度条
        self.progress_scale = tk.Scale(
            root, from_=0, to=100, orient=tk.HORIZONTAL,
            showvalue=True, command=self.on_scale_drag
        )
        self.progress_scale.pack(side=tk.BOTTOM, fill=tk.X)
        self.window_size = 500
        self.data_length = 0

        # 绑定事件
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # 创建右键菜单
        self.context_menu = Menu(root, tearoff=0)
        self.context_menu.add_command(
            label="Visualize Superimposed Videos",
            command=self.superimpose_video
        )
        self.context_menu.add_command(
            label="Delete Segment",
            command=self.delete_segment
        )

        # 创建主工具栏
        self.toolbar = tk.Frame(root)  # 确保这行存在
        self.load_button = tk.Button(
            self.toolbar, text="Load Data", command=self.load_data
        )
        self.load_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(
            self.toolbar, text="Save Segments", command=self.save_segments
        )
        self.save_button.pack(side=tk.LEFT)

        self.threshold_button = tk.Button(
            self.toolbar, text="Adjust Thresholds", command=self.adjust_thresholds
        )
        self.threshold_button.pack(side=tk.LEFT)

        self.viz_button = tk.Button(
            self.toolbar, text="Visualize Settings", command=self.show_viz_settings
        )
        self.viz_button.pack(side=tk.LEFT)

        self.toolbar.pack(side=tk.TOP, fill=tk.X)  # 确保这行存在


        # 初始化颜色设置
        self.line_colors = {
            'eye_left': '#FF0000',
            'eye_right': '#0000FF',
            'tail_angle': '#00FF00',
            'swimbladder_top': '#FFA500',
            'swimbladder_side': '#800080',
            'head_motion': '#008080',
            'jaw_motion': '#FF00FF'  # 新增品红色
        }
        self.color_entries = {}

        self.dragging = None
        self.current_artist = None

    def load_data(self):
        self.father_folder = filedialog.askdirectory(
            title="选择父文件夹",
            mustexist=True
        )
        if not self.father_folder:  # 用户取消选择
            return

        self.base_name = simpledialog.askstring("Data Name", "请输入视频名称（示例：FLIR_2024-08-01_F4_01-T1-0000）:")
        if not self.base_name:
            return

        with open('sleap_ztrack_file_corresponding/sleap_ztrack_file_corresponding.json') as json_data:
            name_dict = json.load(json_data)

        # 配置数据路径
        data_paths = {
            'eye_angle': f'{self.father_folder}/ztrack_results_h5/{name_dict[self.base_name + '.h5']}',
            'tail_angle': f'{self.father_folder}/ztrack_results_h5/{name_dict[self.base_name + '.h5']}',
            'swimbladder_top': f"{self.father_folder}/temp_check_vector_motion/swimbladder_motion/{name_dict[self.base_name + '.h5'].split('.')[0].split('_Trial')[0] + '-T' + name_dict[self.base_name + '.h5'].split('.')[0].split('Trial')[1]}/TOPCAM/y_shift.npy",
            'swimbladder_side': f"{self.father_folder}/temp_check_vector_motion/swimbladder_motion/{name_dict[self.base_name + '.h5'].split('.')[0].split('_Trial')[0] + '-T' + name_dict[self.base_name + '.h5'].split('.')[0].split('Trial')[1]}/SIDECAM/x_shift.npy",
            'head_motion': f'{self.father_folder}/temp_check_vector_motion/distance_data/{self.base_name}/original/l2_norm_4_35.npy',
            'jaw_motion': f'{self.father_folder}/temp_check_vector_motion/displacement_rotation_data/data/{self.base_name}/'
        }
        print('loading data...')

        try:
            for dtype in self.data_types:

                if dtype == 'eye_angle':
                    full_path = data_paths[dtype]
                    if not os.path.exists(full_path):
                        raise FileNotFoundError(f"文件不存在: {full_path}")
                    df_eye = pd.read_hdf(full_path, "eye")
                    eye_angles = df_eye[[("left_eye", "angle"), ("right_eye", "angle")]].values
                    eye_angles_filt = low_pass_filt(eye_angles, 200, 2).T
                    left_angle = eye_angles_filt[0]
                    right_angle = eye_angles_filt[1]

                    self.data[dtype]['left'] = left_angle[:, np.newaxis]
                    self.data[dtype]['right'] = right_angle[:, np.newaxis]

                elif dtype == 'tail_angle':
                    full_path = data_paths[dtype]
                    if not os.path.exists(full_path):
                        raise FileNotFoundError(f"文件不存在: {full_path}")
                    tail_vigor = get_tail_angles(pd.read_hdf(full_path, "tail"),
                                                 pd.read_hdf(full_path, "eye")["heading"].values)
                    tail_vigor = np.mean(tail_vigor, axis=1)
                    self.data[dtype] = tail_vigor

                elif dtype == 'swimbladder_top':
                    full_path = data_paths[dtype]
                    if not os.path.exists(full_path):
                        raise FileNotFoundError(f"文件不存在: {full_path}")
                    self.data[dtype] = np.load(full_path)

                    swim_bladder_events_top = np.load(
                        f"{self.father_folder}/temp_check_vector_motion/swimbladder_motion/{name_dict[self.base_name + '.h5'].split('.')[0].split('_Trial')[0] + '-T' + name_dict[self.base_name + '.h5'].split('.')[0].split('Trial')[1]}/TOPCAM/episodes_idx.npy")
                    swim_bladder_events_top = combine_episodes(swim_bladder_events_top)
                    self.segments[dtype] = swim_bladder_events_top

                elif dtype == 'swimbladder_side':
                    full_path = data_paths[dtype]
                    if not os.path.exists(full_path):
                        raise FileNotFoundError(f"文件不存在: {full_path}")
                    self.data[dtype] = np.load(full_path)
                    swim_bladder_events_side = np.load(
                        f"{self.father_folder}/temp_check_vector_motion/swimbladder_motion/{name_dict[self.base_name + '.h5'].split('.')[0].split('_Trial')[0] + '-T' + name_dict[self.base_name + '.h5'].split('.')[0].split('Trial')[1]}/SIDECAM/episodes_idx.npy")

                    swim_bladder_events_side = combine_episodes(swim_bladder_events_side)
                    self.segments[dtype] = swim_bladder_events_side

                elif dtype == 'head_motion':
                    full_path = data_paths[dtype]
                    if not os.path.exists(full_path):
                        raise FileNotFoundError(f"文件不存在: {full_path}")

                    self.data[dtype] = np.load(full_path)

                elif dtype == 'jaw_motion':
                    full_path = data_paths[dtype]
                    if not os.path.exists(full_path):
                        raise FileNotFoundError(f"文件不存在: {full_path}")
                    jaw_front_motion_x = np.load(
                        f'{full_path}/jaw_front_motion_x.npy')
                    jaw_front_motion_y = np.load(
                        f'{full_path}/jaw_front_motion_y.npy')
                    all = np.sqrt(jaw_front_motion_x ** 2 + jaw_front_motion_y ** 2)
                    all[0] = np.mean(all[1:])
                    print('jaw shape:', all.shape)
                    self.data[dtype] = all

                # 初始自动计算阈值和分段
                self.initial_calculation(dtype)
                self.update_threshold_line(dtype)
                self.initial_thresholds['tail_angle'] = self.thresholds['tail_angle']
                self.initial_thresholds['head_motion'] = self.thresholds['head_motion']
                self.initial_thresholds['jaw_motion'] = self.thresholds['jaw_motion']

            self.update_visualization()
            self.canvas.draw_idle()
            # 在try块的末尾，数据加载完成后添加：
            self.data_length = self.data['eye_angle']['left'].shape[0]
            self.progress_scale.config(from_=0,
                                       to=self.data_length - self.window_size)
            self.progress_scale.set(0)
            self._update_axes_xlim(0, self.window_size)
        except Exception as e:
            traceback.print_exc()  # 打印完整错误信息
            messagebox.showerror("加载错误", f"数据加载失败: {str(e)}")
            self.reset_state()  # 新增重置状态方法

    def reset_state(self):
        """加载失败时重置所有状态"""
        self.data = {t: None for t in self.data_types}
        self.segments = {t: [] for t in self.data_types}
        self.thresholds = {t: None for t in self.data_types}
        self.progress_scale.set(0)
        self.canvas.draw_idle()

    def _update_axes_xlim(self, start, end):
        """统一更新所有子图的x轴范围"""
        for ax in self.axes.values():
            ax.set_xlim(start, end)
        self.canvas.draw_idle()

    def on_scale_drag(self, value):
        """处理进度条拖动事件"""
        try:
            start = int(float(value))
            end = start + self.window_size
            self._update_axes_xlim(start, end)
        except Exception as e:
            print("Error updating x range:", e)

    def _update_window_size(self, new_size, center=None):
        """更新可视窗口大小并保持中心位置"""
        if center is None:
            current_start = self.progress_scale.get()
            center = current_start + self.window_size / 2

        self.window_size = max(100, min(new_size, self.data_length))
        self.progress_scale.config(to=self.data_length - self.window_size)

        new_start = int(center - self.window_size / 2)
        new_start = max(0, min(new_start, self.data_length - self.window_size))
        self.progress_scale.set(new_start)
        self.on_scale_drag(new_start)

    def initial_calculation(self, dtype):
        """初始自动计算阈值和分段"""
        print('Initial calculation')
        if dtype == 'eye_angle':
            self.segments[dtype] = self.calculate_eye()
        elif dtype == 'tail_angle':
            self.thresholds[dtype], self.segments[dtype] = self.calculate_tail()
        elif dtype == 'head_motion':
            self.thresholds[dtype], self.segments[dtype] = self.calculate_head_motion()
        elif dtype == 'jaw_motion':
            self.thresholds[dtype], self.segments[dtype] = self.calculate_jaw_motion()

    def update_threshold_line(self, dtype):
        """更新阈值线显示"""
        if dtype in self.manual_dtypes:
            line = self.threshold_lines[dtype]
            line.set_xdata([0, self.data[dtype].shape[0]])
            line.set_ydata([self.thresholds[dtype]] * 2)
            line.set_visible(True)

    def adjust_thresholds(self):
        """弹出阈值调整对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("调整阈值")
        dialog.geometry("350x260")  # 增加高度

        # 获取初始阈值
        components = [
            ('tail_angle', 'Tail bout'),
            ('head_motion', 'Head Motion'),
            ('jaw_motion', 'Jaw Movement')  # 新增项
        ]

        entries = {}
        for row, (dtype, label) in enumerate(components):
            # 初始值显示
            initial = self.initial_thresholds.get(dtype, 'N/A')
            tk.Label(dialog, text=f"{label} Threshold:",
                     font=('Arial', 10)).grid(row=row * 3, column=0, padx=5, sticky='w')
            tk.Label(dialog, text=f"初始值: {initial:.2f}",
                     font=('Arial', 8), fg='gray').grid(row=row * 3 + 1, column=0, padx=5, sticky='w')

            # 输入框
            entry = tk.Entry(dialog)
            entry.insert(0, f"{self.thresholds[dtype]:.2f}")
            entry.grid(row=row * 3 + 2, column=0, padx=5, pady=2)
            entries[dtype] = entry

        # 应用按钮
        def apply_changes():
            try:
                for dtype, entry in entries.items():
                    new_val = float(entry.get())
                    self._update_threshold(dtype, new_val)

                dialog.destroy()
                self.update_visualization()
                self.canvas.draw_idle()
            except ValueError:
                messagebox.showerror("输入错误", "请输入有效的数字")

        tk.Button(dialog, text="应用", command=apply_changes).grid(row=len(components) * 3, columnspan=2, pady=10)

    def _update_threshold(self, dtype, new_value):
        """通用阈值更新方法"""
        if dtype not in self.manual_dtypes:
            return

        # 更新阈值线
        self.thresholds[dtype] = new_value
        self.threshold_lines[dtype].set_ydata([new_value] * 2)

        # 重新计算segments
        if dtype == 'tail_angle':
            _, self.segments[dtype] = self.calculate_tail(t=new_value)
        elif dtype == 'head_motion':
            _, self.segments[dtype] = self.calculate_head_motion(t=new_value)
        elif dtype == 'jaw_motion':  # 新增jaw_motion处理
            _, self.segments[dtype] = self.calculate_jaw_motion(t=new_value)

    def calculate_eye(self):
        print('Calculating eye convergence...')

        left = self.data['eye_angle']['left']
        right = self.data['eye_angle']['right']
        data = left + right

        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(data)

        # Get the means of the components
        means = gmm.means_.flatten()
        higher_mean_index = np.argmax(means)

        # Calculate probabilities for the original data points under the higher mean component
        probabilities_higher_mean = gmm.predict_proba(data)[:, higher_mean_index]
        mov_means = moving_average(probabilities_higher_mean)
        # print(mov_means.shape)

        valid_periods = find_valid_periods(mov_means)
        # for lists in valid_periods:
        #     print(lists.shape)
        print('Total Valid Periods: ', len(valid_periods))

        return valid_periods

    def calculate_tail(self, t=None):
        print('Calculating tail bouts...')
        tail_angles = self.data['tail_angle'][:, np.newaxis]
        l2_norms = np.linalg.norm(tail_angles, ord=2, axis=1)
        print('l2_norms.shape:', l2_norms.shape)

        std_dev = 4
        kernel_size = 35  # Adjust size as needed
        bw_method = 0.05
        gaussian_kernel = (
                np.exp(-0.5 * (np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size) / std_dev) ** 2) /
                np.sum(np.exp(
                    -0.5 * (np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size) / std_dev) ** 2)))

        # Convolve the derivative with the Gaussian kernel
        convolved_result = np.convolve(l2_norms, gaussian_kernel, mode='same')
        convolved_result[0] = convolved_result[4]

        if t is None:

            # Create a Gaussian kernel with standard deviation
            kde = gaussian_kde(convolved_result, bw_method=bw_method)
            # Create a range of values to evaluate the KDE
            x = np.linspace(0, np.max(convolved_result), 5000)  # Adjust range as needed

            # Evaluate the KDE on the range of values
            kde_values = kde(x)
            kde_minima = argrelextrema(kde_values, np.less)[0]

            # Check for bimodality by finding peaks in the KDE
            peaks, _ = find_peaks(kde_values, height=0)  # Find peaks in the KDE

            inflection_indices = kde_minima[2]
            antimode_value = x[inflection_indices]
        else:
            print(f'Updating tail bouts, new antimode value: {t}')
            antimode_value = t

        # delete peaks that are below the threshold.
        peaks_data, _ = find_peaks(convolved_result)
        peaks, _ = find_peaks(convolved_result)
        peaks_data = np.delete(peaks_data, np.argwhere(convolved_result[peaks_data] < antimode_value), axis=0)

        # Find local minima and delete those above 0.
        local_minima = argrelextrema(convolved_result, np.less)[0]
        local_minima = np.array([num for num in local_minima if convolved_result[num] < antimode_value])

        peak_to_delete = []
        index_del = 0
        stop_point_of_peak_sifting = peaks_data[peaks_data < local_minima[-1]].shape[0] - 1
        peaks_data = peaks_data[: stop_point_of_peak_sifting + 1]

        while index_del <= stop_point_of_peak_sifting:
            if index_del == 0:
                larger_numbers_past = local_minima[local_minima < peaks_data[index_del]]
                if np.any(larger_numbers_past):
                    index_del += 1
                else:
                    peak_to_delete.append(index_del)
                    index_del += 1
            elif index_del <= stop_point_of_peak_sifting:
                larger_numbers_past = local_minima[
                    (peaks_data[index_del - 1] < local_minima) & (local_minima < peaks_data[index_del])]
                if np.any(larger_numbers_past):
                    index_del += 1
                else:
                    peak_to_delete.append(index_del)
                    index_del += 1

        peaks_data = np.delete(peaks_data, peak_to_delete, axis=0)

        bouts = []
        length_bouts = []
        for i in range(peaks_data.shape[0]):
            if i == 0:
                nearest_minima = find_nearest_bigger(peaks_data[i], peaks_data[i + 1], local_minima, convolved_result,
                                                     antimode_value)
                start = find_start_point(0, peaks_data[i], convolved_result, antimode_value)
                bouts.append([start, nearest_minima])
                length_bouts.append(nearest_minima - peaks_data[i])
            elif i < peaks_data.shape[0] - 1:
                nearest_minima = find_nearest_bigger(peaks_data[i], peaks_data[i + 1], local_minima, convolved_result,
                                                     antimode_value)
                start = find_start_point(peaks_data[i - 1], peaks_data[i], convolved_result, antimode_value)
                bouts.append([start, nearest_minima])
                length_bouts.append(nearest_minima - peaks_data[i])
            else:
                nearest_minima = find_nearest_bigger(peaks_data[i], convolved_result.shape[0], local_minima,
                                                     convolved_result, antimode_value)
                start = find_start_point(peaks_data[i - 1], peaks_data[i], convolved_result, antimode_value)
                bouts.append([start, nearest_minima])
                length_bouts.append(nearest_minima - peaks_data[i])

        print(f"Number of the tail bouts: {len(length_bouts)}")
        print('Average Length of the tail bouts:', np.mean(length_bouts))

        # for accidentally overlapping bouts.
        for i in range(1, len(bouts)):
            if bouts[i - 1][1] >= bouts[i][0]:
                bouts[i - 1][1] = bouts[i][0] - 1

        return antimode_value, bouts

    def calculate_head_motion(self, t=None):
        print('Calculating head motion events...')

        all = self.data['head_motion']

        std_dev_d = 10
        kernel_size_d = 50  # Adjust size as needed

        gaussian_kernel = (np.exp(
            -0.5 * (np.linspace(-kernel_size_d // 2, kernel_size_d // 2, kernel_size_d) / std_dev_d) ** 2) /
                           np.sum(np.exp(
                               -0.5 * (np.linspace(-kernel_size_d // 2, kernel_size_d // 2,
                                                   kernel_size_d) / std_dev_d) ** 2)))

        # Convolve the derivative with the Gaussian kernel
        convolved_result = np.convolve(all, gaussian_kernel, mode='same')

        if t is None:

            x = np.linspace(0, np.max(convolved_result), 5000)[:, newaxis]  # Adjust range as needed

            gmm = GaussianMixture(n_components=20, random_state=0)
            gmm.fit(x)

            # Calculate the PDF using the GMM
            pdf = np.exp(gmm.score_samples(x))

            # Find the local minima in the PDF
            # Note: To find minima, we can use the negative of the PDF
            neg_pdf = -pdf
            peaks, _ = find_peaks(neg_pdf)  # Find peaks in the negative PDF, which correspond to minima in the original PDF

            # Find the index of the first local minimum
            if len(peaks) > 0:
                first_local_min_index = peaks[0]
                antimode_value = x[first_local_min_index][0]
            else:
                print("No local minima found.")
        else:
            print(f'Updating head motion events, new antimode value: {t}')
            antimode_value = t

        # delete peaks that are below the threshold.
        peaks_data, _ = find_peaks(convolved_result)
        peaks_data = np.delete(peaks_data, np.argwhere(convolved_result[peaks_data] < antimode_value), axis=0)

        # Find local minima and delete those above 0.
        local_minima = argrelextrema(convolved_result, np.less)[0]
        local_minima = np.array([num for num in local_minima if convolved_result[num] < antimode_value])
        # local_minima = np.array([num for num in local_minima if convolved_result[num] < cut_off])

        peak_to_delete = []
        index_del = 0
        stop_point_of_peak_sifting = peaks_data[peaks_data < local_minima[-1]].shape[0] - 1
        peaks_data = peaks_data[: stop_point_of_peak_sifting + 1]

        while index_del <= stop_point_of_peak_sifting:
            if index_del == 0:
                larger_numbers_past = local_minima[local_minima < peaks_data[index_del]]
                if np.any(larger_numbers_past):
                    index_del += 1
                else:
                    peak_to_delete.append(index_del)
                    index_del += 1
            elif index_del <= stop_point_of_peak_sifting:
                larger_numbers_past = local_minima[
                    (peaks_data[index_del - 1] < local_minima) & (local_minima < peaks_data[index_del])]
                if np.any(larger_numbers_past):
                    index_del += 1
                else:
                    peak_to_delete.append(index_del)
                    index_del += 1

        peaks_data = np.delete(peaks_data, peak_to_delete, axis=0)

        bouts = []
        # length_bouts = []
        for i in range(peaks_data.shape[0]):
            if i == 0:
                nearest_minima = find_nearest_bigger_v3(peaks_data[i], peaks_data[i + 1], local_minima, convolved_result,
                                                     antimode_value)
                start = find_start_point_v2(0, peaks_data[i], local_minima)
                bouts.append([start, nearest_minima])
                # length_bouts.append(nearest_minima - peaks_data[i])
            elif i < peaks_data.shape[0] - 1:
                nearest_minima = find_nearest_bigger_v3(peaks_data[i], peaks_data[i + 1], local_minima, convolved_result,
                                                     antimode_value)
                start = find_start_point_v2(peaks_data[i - 1], peaks_data[i], local_minima)
                bouts.append([start, nearest_minima])
                # length_bouts.append(nearest_minima-peaks_data[i])
            else:
                nearest_minima = find_nearest_bigger_v3(peaks_data[i], convolved_result.shape[0], local_minima,
                                                     convolved_result, antimode_value)
                start = find_start_point_v2(peaks_data[i - 1], peaks_data[i], local_minima)
                bouts.append([start, nearest_minima])
                # length_bouts.append(nearest_minima-peaks_data[i])

        print(f"Number of the bouts: {len(bouts)}")
        # print('Average Length of the bouts:', np.mean(length_bouts))

        # for accidentally overlapping bouts.
        for i in range(1, len(bouts)):
            if bouts[i - 1][1] >= bouts[i][0]:
                bouts[i - 1][1] = bouts[i][0] - 1

        return antimode_value, bouts

    def calculate_jaw_motion(self, t=None):
        all = self.data['jaw_motion']

        std_dev_d = 10
        kernel_size_d = 50  # Adjust size as needed
        bw_method = 0.025

        # derivatives = np.diff(l2_norms, axis=0)
        gaussian_kernel = (
                np.exp(-0.5 * (np.linspace(-kernel_size_d // 2, kernel_size_d // 2, kernel_size_d) / std_dev_d) ** 2) /
                np.sum(np.exp(
                    -0.5 * (np.linspace(-kernel_size_d // 2, kernel_size_d // 2, kernel_size_d) / std_dev_d) ** 2)))

        # Convolve the derivative with the Gaussian kernel
        convolved_result = np.convolve(all, gaussian_kernel, mode='same')

        if t is None:
            # Create a Gaussian kernel with standard deviation
            kde = gaussian_kde(convolved_result, bw_method=bw_method)
            # Create a range of values to evaluate the KDE
            x = np.linspace(0, np.max(convolved_result), 5000)  # Adjust range as needed

            # Evaluate the KDE on the range of values
            kde_values = kde(x)
            kde_minima = argrelextrema(kde_values, np.less)[0]

            # Check for bimodality by finding peaks in the KDE
            peaks, _ = find_peaks(kde_values, height=0)  # Find peaks in the KDE
            num_peaks = len(peaks)

            if num_peaks == 2:
                print(f"The distribution is bimodal")
                inflection_indices = kde_minima[0]
                antimode_value = x[inflection_indices]
                print(f"The distribution's antimode value is {antimode_value}")
            elif num_peaks > 2:
                print(f"The distribution is multimodal")
                inflection_indices = kde_minima[0]
                antimode_value = x[inflection_indices]
                print(f"The distribution is multimodal. Number of peaks: {num_peaks}")
            else:
                print(f"The distribution is not bimodal. Number of peaks: {num_peaks}")
                exit()
        else:
            antimode_value = t

        # delete peaks that are below the threshold.
        peaks_data, _ = find_peaks(convolved_result)
        peaks_data = np.delete(peaks_data, np.argwhere(convolved_result[peaks_data] < antimode_value), axis=0)

        # Find local minima and delete those above 0.
        local_minima = argrelextrema(convolved_result, np.less)[0]
        local_minima = np.array([num for num in local_minima if convolved_result[num] < antimode_value])
        # local_minima = np.array([num for num in local_minima if convolved_result[num] < cut_off])

        peak_to_delete = []
        index_del = 0
        stop_point_of_peak_sifting = peaks_data[peaks_data < local_minima[-1]].shape[0] - 1
        peaks_data = peaks_data[: stop_point_of_peak_sifting + 1]

        while index_del <= stop_point_of_peak_sifting:
            if index_del == 0:
                larger_numbers_past = local_minima[local_minima < peaks_data[index_del]]
                if np.any(larger_numbers_past):
                    index_del += 1
                else:
                    peak_to_delete.append(index_del)
                    index_del += 1
            elif index_del <= stop_point_of_peak_sifting:
                larger_numbers_past = local_minima[
                    (peaks_data[index_del - 1] < local_minima) & (local_minima < peaks_data[index_del])]
                if np.any(larger_numbers_past):
                    index_del += 1
                else:
                    peak_to_delete.append(index_del)
                    index_del += 1

        peaks_data = np.delete(peaks_data, peak_to_delete, axis=0)

        bouts = []
        # length_bouts = []
        for i in range(peaks_data.shape[0]):
            if i == 0:
                nearest_minima = find_nearest_bigger(peaks_data[i], peaks_data[i + 1], local_minima, convolved_result,
                                                     antimode_value)
                start = find_start_point(0, peaks_data[i], convolved_result, antimode_value)
                bouts.append([start, nearest_minima])
                # length_bouts.append(nearest_minima - peaks_data[i])
            elif i < peaks_data.shape[0] - 1:
                nearest_minima = find_nearest_bigger(peaks_data[i], peaks_data[i + 1], local_minima, convolved_result,
                                                     antimode_value)
                start = find_start_point(peaks_data[i - 1], peaks_data[i], convolved_result, antimode_value)
                bouts.append([start, nearest_minima])
                # length_bouts.append(nearest_minima-peaks_data[i])
            else:
                nearest_minima = find_nearest_bigger(peaks_data[i], convolved_result.shape[0], local_minima,
                                                     convolved_result, antimode_value)
                start = find_start_point(peaks_data[i - 1], peaks_data[i], convolved_result, antimode_value)
                bouts.append([start, nearest_minima])
                # length_bouts.append(nearest_minima-peaks_data[i])

        print(f"Number of the jaw bouts: {len(bouts)}")

        # for accidentally overlapping bouts.
        for i in range(1, len(bouts)):
            if bouts[i - 1][1] >= bouts[i][0]:
                bouts[i - 1][1] = bouts[i][0] - 1

        return antimode_value, bouts


    def update_visualization(self):
        print("Updating visualization")

        # 清除之前的注释
        if self.segment_annotation:
            self.segment_annotation.remove()
            self.segment_annotation = None

        for dtype in self.data_types:
            ax = self.axes[dtype]

            # 清除非原始元素
            for artist in ax.lines[2:]:
                artist.remove()
            for patch in ax.patches:
                patch.remove()

            # 更新数据线
            if dtype == 'eye_angle' and self.data[dtype]['left'] is not None:
                self.data_lines[dtype]['left'].set_data(
                    np.arange(len(self.data[dtype]['left'])),
                    self.data[dtype]['left']
                )
                self.data_lines[dtype]['right'].set_data(
                    np.arange(len(self.data[dtype]['right'])),
                    self.data[dtype]['right']
                )
            elif self.data[dtype] is not None:
                self.data_lines[dtype].set_data(
                    np.arange(len(self.data[dtype])),
                    self.data[dtype]
                )
            ax.relim()
            ax.autoscale_view()

            # 绘制segments（新增颜色判断）
            for idx, (start, end) in enumerate(self.segments[dtype]):
                # 边界线
                ax.axvline(start, color='red', linestyle='--', alpha=0.7)
                ax.axvline(end, color='blue', linestyle='--', alpha=0.7)

                # 根据选中状态设置颜色
                fill_color = 'lime' if (dtype, idx) == self.selected_segment else 'grey'

                # 矩形填充
                ax.add_patch(Rectangle(
                    (start, ax.get_ylim()[0]),
                    end - start,
                    ax.get_ylim()[1] - ax.get_ylim()[0],
                    alpha=0.2,
                    color=fill_color
                ))

                # 如果是选中的segment，添加注释显示start和end值
                if (dtype, idx) == self.selected_segment:
                    # 获取当前视图范围
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    
                    # 计算注释位置（在segment中间偏上的位置）
                    x_pos = (start + end) / 2
                    y_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.8  # 在y轴80%的位置
                    
                    # 添加注释
                    self.segment_annotation = ax.annotate(
                        f"Start: {int(start)}, End: {int(end)}",
                        xy=(x_pos, y_pos),
                        xytext=(0, 10),  # 文本偏移
                        textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
                        ha='center'
                    )

        self.canvas.draw_idle()

    # 事件处理函数
    def on_scroll(self, event):
        """处理滚动事件，按住Shift时缩放窗口大小"""
        if event.key == 'shift':
            base_scale = 1.1
            scale_factor = 1 / base_scale if event.button == 'up' else base_scale
            xdata = event.xdata if event.xdata else self.progress_scale.get() + self.window_size / 2

            new_window_size = int(self.window_size * scale_factor)
            self._update_window_size(new_window_size, xdata)

    def on_pick(self, event):
        if isinstance(event.artist, Line2D) and event.artist in self.threshold_lines.values():
            self.current_artist = event.artist
            self.dragging = True

    def on_motion(self, event):
        """处理鼠标移动事件"""
        if self.dragging and self.current_artist and event.inaxes:
            dtype = self.current_artist
            if dtype in self.manual_dtypes:
                # 更新阈值线位置
                new_threshold = event.ydata
                self.thresholds[dtype] = new_threshold
                self.threshold_lines[dtype].set_ydata([new_threshold] * 2)

                # 重新计算相关数据段
                if dtype == 'tail_angle':
                    _, self.segments[dtype] = self.calculate_tail(t=new_threshold)
                elif dtype == 'head_motion':
                    _, self.segments[dtype] = self.calculate_head_motion(t=new_threshold)

                self.update_visualization()
                self.canvas.draw_idle()

    def on_release(self, event):
        """处理鼠标释放事件"""
        self.dragging = False
        self.current_artist = None

    def on_click(self, event):
        """处理鼠标点击事件（修改后的版本）"""
        if event.button == 3:  # 右键菜单
            self.context_menu.tk_popup(event.x, event.y)
        elif event.button == 1:  # 左键处理
            if event.inaxes:
                # Shift+左键拖动阈值线（原有逻辑）
                if event.key == 'shift':
                    for dtype in self.manual_dtypes:
                        ax = self.axes[dtype]
                        if ax == event.inaxes:
                            y_threshold = self.thresholds[dtype]
                            if y_threshold is None:
                                continue
                            if abs(event.ydata - y_threshold) < 0.5:
                                self.dragging = True
                                self.current_artist = dtype
                                return
                # 正常左键选择
                else:
                    old_selection = self.selected_segment
                    self.selected_segment = None  # 先重置选中状态
                    
                    for dtype, ax in self.axes.items():
                        if ax == event.inaxes:
                            x = event.xdata
                            # 精确查找匹配的segment
                            for idx, (start, end) in enumerate(self.segments[dtype]):
                                if start <= x <= end:
                                    self.selected_segment = (dtype, idx)
                                    break  # 找到第一个匹配项即退出
                            if self.selected_segment:
                                break  # 找到匹配项后退出循环

                    # 只有当选择改变时才刷新显示
                    if old_selection != self.selected_segment:
                        self.update_visualization()
                        self.canvas.draw_idle()

    def delete_segment(self):
        """删除选中段落并记录操作历史"""
        if self.selected_segment:
            dtype, idx = self.selected_segment
            try:
                # 记录被删除的段落信息
                deleted_segment = {
                    'dtype': dtype,
                    'index': idx,
                    'segment': self.segments[dtype][idx].copy()
                }
                self.undo_stack.append(deleted_segment)

                # 执行删除操作
                del self.segments[dtype][idx]
                self.update_visualization()

            except IndexError:
                messagebox.showerror("错误", "片段不存在")
            finally:
                self.selected_segment = None

    def undo_last_delete(self, event=None):
        """撤销最后一次删除操作"""
        if not self.undo_stack:
            messagebox.showinfo("提示", "没有可撤销的操作")
            return

        # 获取最后删除的记录
        last_deleted = self.undo_stack.pop()

        # 恢复数据
        dtype = last_deleted['dtype']
        idx = last_deleted['index']
        segment = last_deleted['segment']

        try:
            # 插入到原始位置
            self.segments[dtype].insert(idx, segment)
            self.update_visualization()
            messagebox.showinfo("撤销成功", "已恢复最后删除的段落")
        except Exception as e:
            messagebox.showerror("撤销失败", str(e))

    def save_segments(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            try:
                # 安全转换数据格式
                serializable = {}
                for dtype, segments in self.segments.items():
                    serializable[dtype] = []
                    for seg in segments:
                        # 处理numpy数组等特殊类型
                        if hasattr(seg, 'tolist'):
                            seg = seg.tolist()

                        if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                            try:
                                s = float(seg[0])
                                e = float(seg[1])
                                serializable[dtype].append([s, e])
                            except (TypeError, ValueError):
                                print(f"无法转换分段值：{dtype} - {seg}")
                        else:
                            print(f"忽略无效分段格式：{dtype} - {seg}")

                # 保存数据
                with open(file_path, 'w') as f:
                    json.dump(serializable, f, indent=2)

                messagebox.showinfo("成功", f"数据已保存到：\n{file_path}")

            except Exception as e:
                messagebox.showerror("保存失败", f"错误详情：\n{str(e)}")

    def superimpose_video(self):
        """视频叠加入口函数"""
        if not self.selected_segment:
            messagebox.showerror("错误", "请先选择要可视化的段落")
            return

        # 获取选中段落信息
        dtype, idx = self.selected_segment
        try:
            start, end = self.segments[dtype][idx]
        except IndexError:
            messagebox.showerror("错误", "选择的段落不存在")
            return

        frame_folder = filedialog.askdirectory(
            title="选择视频帧路径",
            mustexist=True
        )
        if not frame_folder:  # 用户取消选择
            return

        # 弹出目录选择对话框
        save_dir = filedialog.askdirectory(
            title="选择视频保存目录",
            mustexist=True
        )
        if not save_dir:  # 用户取消选择
            return

        # 调用视频生成函数
        self.superimpose_video_clipper(frame_folder, save_dir, start, end, self.father_folder)


    def superimpose_video_clipper(self, frame_folder, save_path, start, end, father_folder):
        superimpose_annotation(self.base_name, ranges=[start, end], output_path=save_path, frame_path=frame_folder, father_path=father_folder)

    def show_viz_settings(self):
        """显示可视化设置窗口"""
        settings_win = tk.Toplevel(self.root)
        settings_win.title("可视化设置")
        settings_win.geometry("400x450")  # 增加高度

        # 创建颜色设置区域
        color_frame = tk.LabelFrame(settings_win, text="线条颜色设置")
        color_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)


        color_items = [
            ("Eye Left", 'eye_left'),
            ("Eye Right", 'eye_right'),
            ("Tail Angle", 'tail_angle'),
            ("Swimbladder Top", 'swimbladder_top'),
            ("Swimbladder Side", 'swimbladder_side'),
            ("Head Motion", 'head_motion'),
            ("Jaw Movement", 'jaw_motion')  # 新增项
        ]

        # 创建颜色选择行
        for idx, (display_name, color_key) in enumerate(color_items):
            row = tk.Frame(color_frame)
            row.pack(pady=3, fill=tk.X)

            # 颜色选择按钮
            btn = tk.Button(
                row,
                text=display_name,
                width=15,
                command=lambda k=color_key: self.choose_color(k)
            )
            btn.pack(side=tk.LEFT, padx=5)

            # 颜色预览
            preview = tk.Label(row, bg=self.line_colors[color_key], width=5)
            preview.pack(side=tk.LEFT)

            # RGB输入框
            entry = tk.Entry(row, width=15)
            entry.insert(0, self.color_to_rgb_str(self.line_colors[color_key]))
            entry.pack(side=tk.LEFT, padx=5)
            self.color_entries[color_key] = (preview, entry)

        # 应用按钮
        apply_btn = tk.Button(
            settings_win,
            text="应用更改",
            command=self.apply_color_changes
        )
        apply_btn.pack(pady=10)

    def choose_color(self, color_key):
        """打开颜色选择对话框（修正后的版本）"""
        color = colorchooser.askcolor(  # 修改这行调用方式
            title=f"选择{color_key}颜色",
            initialcolor=self.line_colors[color_key]
        )
        if color and color[1]:  # 添加空值检查
            rgb_str = self.color_to_rgb_str(color[1])
            preview, entry = self.color_entries[color_key]
            preview.config(bg=color[1])
            entry.delete(0, tk.END)
            entry.insert(0, rgb_str)

    def apply_color_changes(self):
        """应用颜色修改到图表"""
        for color_key, (preview, entry) in self.color_entries.items():
            try:
                # 转换RGB字符串到十六进制
                rgb_str = entry.get()
                hex_color = self.rgb_str_to_hex(rgb_str)

                # 更新颜色配置
                self.line_colors[color_key] = hex_color
                preview.config(bg=hex_color)

                # 更新图表线条
                self.update_line_color(color_key, hex_color)

            except ValueError:
                messagebox.showerror("错误", f"{color_key}颜色值无效")

        self.canvas.draw_idle()

    def update_line_color(self, color_key, hex_color):
        """更新具体线条颜色"""
        if color_key == 'eye_left':
            self.data_lines['eye_angle']['left'].set_color(hex_color)
        elif color_key == 'eye_right':
            self.data_lines['eye_angle']['right'].set_color(hex_color)
        else:
            dtype = color_key if color_key in self.data_types else {
                'tail_angle': 'tail_angle',
                'swimbladder_top': 'swimbladder_top',
                'swimbladder_side': 'swimbladder_side',
                'head_motion': 'head_motion'
            }[color_key]
            self.data_lines[dtype].set_color(hex_color)

    # 颜色转换工具方法
    def color_to_rgb_str(self, hex_color):
        """十六进制颜色转RGB字符串"""
        rgb = tuple(int(hex_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

    def rgb_str_to_hex(self, rgb_str):
        """RGB字符串转十六进制"""
        rgb = [int(x) for x in rgb_str[4:-1].split(',')]
        return "#{:02x}{:02x}{:02x}".format(*rgb)

if __name__ == "__main__":
    root = tk.Tk()
    app = DataVisualizerApp(root)
    root.mainloop()