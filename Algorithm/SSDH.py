import os
from collections import defaultdict, deque
import numpy as np
import networkx as nx
import random
import time
import heapq
import json 
import csv
from datetime import datetime


def save_precomputed_costs(data_to_save, filename):
    print(f"--- 正在将预计算结果保存到 '{filename}' ---")

    serializable_data = {
        'node_hyperdegrees': data_to_save['node_hyperdegrees'],
        'all_pairs_info_dist': {k: dict(v) for k, v in data_to_save['all_pairs_info_dist'].items()},
        'all_pairs_hop_count': {k: dict(v) for k, v in data_to_save['all_pairs_hop_count'].items()}
    }

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=4)
        print("保存成功。")
    except Exception as e:
        print(f"错误：保存文件失败。原因: {e}")

def read_hypergraph_data(file_path):
    try:
        with open(file_path, 'r') as file:
            hyper_edges = [list(map(int, line.strip().split())) for line in file if line.strip()]
        print(f"成功从 {file_path} 加载了 {len(hyper_edges)} 条超边。")
        return hyper_edges
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到。")
        return None
    
def load_precomputed_costs(filename):
    print(f"\n--- 发现预计算文件，正在从 '{filename}' 加载 ---")
    start_time = time.time()
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        node_hyperdegrees = {int(k): v for k, v in loaded_data['node_hyperdegrees'].items()}

        all_pairs_info_dist = defaultdict(lambda: defaultdict(lambda: float('inf')))
        for source_str, targets in loaded_data['all_pairs_info_dist'].items():
            for target_str, dist in targets.items():
                all_pairs_info_dist[int(source_str)][int(target_str)] = dist

        all_pairs_hop_count = defaultdict(lambda: defaultdict(lambda: float('inf')))
        for source_str, targets in loaded_data['all_pairs_hop_count'].items():
            for target_str, hops in targets.items():
                all_pairs_hop_count[int(source_str)][int(target_str)] = hops

        end_time = time.time()
        print(f"加载完成，耗时 {end_time - start_time:.2f} 秒。")

        return node_hyperdegrees, all_pairs_info_dist, all_pairs_hop_count

    except Exception as e:
        print(f"错误：加载文件失败或文件格式不正确。原因: {e}")
        return None, None, None


def build_underlying_graph(hyperedges, all_nodes_list):
    g_underlying = nx.Graph()
    g_underlying.add_nodes_from(all_nodes_list)
    for hyperedge in hyperedges:
        nodes_in_edge = [node for node in hyperedge if node in all_nodes_list]
        for i in range(len(nodes_in_edge)):
            for j in range(i + 1, len(nodes_in_edge)):
                u, v = nodes_in_edge[i], nodes_in_edge[j]
                g_underlying.add_edge(u, v)
    return g_underlying

def _single_lineage_traceback(start_node, steps, hyper_edges, node_to_hyperedges_map):
    if steps <= 0: return {start_node}
    current_wave = {start_node}
    for _ in range(steps):
        next_wave = set()
        for node in current_wave:
            if node in node_to_hyperedges_map:
                for he_idx in node_to_hyperedges_map[node]:
                    for neighbor in hyper_edges[he_idx]:
                        if neighbor != node:
                            next_wave.add(neighbor)
        if not next_wave: return current_wave
        current_wave = next_wave
        current_wave.add(start_node)
    return current_wave

def deploy_sensors_global_contribution(all_nodes, total_sensor_num, node_to_hyperedges_map, hyper_edges):
    import math
    sensors = []
    sensor_set = set()

    hyperedge_cover_count = [0] * len(hyper_edges)

    node_contribution = defaultdict(float)

    for _ in range(total_sensor_num):
        best_node, best_gain = None, -1

        for node in all_nodes:
            if node in sensor_set:
                continue

            gain = 0.0
            for he_idx in node_to_hyperedges_map[node]:
                k_prev = hyperedge_cover_count[he_idx]
                k_new = k_prev + 1
                gain += math.log(1 + k_new) - math.log(1 + k_prev)

            if gain > best_gain:
                best_gain = gain
                best_node = node

        if best_node is None:
            break  

        # 确认选择best_node
        sensors.append(best_node)
        sensor_set.add(best_node)

        for he_idx in node_to_hyperedges_map[best_node]:
            hyperedge_cover_count[he_idx] += 1
            node_contribution[best_node] += math.log(1 + hyperedge_cover_count[he_idx]) - math.log(
                hyperedge_cover_count[he_idx]
            )

    print(f"部署完成，选择了 {len(sensors)} 个传感器。")
    return sensors

def generate_candidates_v4(earliest_sensors, t_min, hyper_edges, node_to_hyperedges_map):
    infector_set = {info[1] for _, info in earliest_sensors if info[1] != "unknown"}
    if not infector_set: return set()

    if t_min == 2:
        first_node = next(iter(infector_set))
        if first_node not in node_to_hyperedges_map: return set()
        common_hyperedge_indices = set(node_to_hyperedges_map[first_node])
        for node in list(infector_set)[1:]:
            if node not in node_to_hyperedges_map: return set()
            common_hyperedge_indices.intersection_update(node_to_hyperedges_map[node])
        candidate_sources = set()
        if common_hyperedge_indices:
            for idx in common_hyperedge_indices:
                candidate_sources.update(node for node in hyper_edges[idx] if node not in sensor_list)
        return candidate_sources
    elif t_min >= 3:
        traceback_steps = t_min - 1
        lineage_ancestor_sets = []
        for infector_node in infector_set:
            ancestors = _single_lineage_traceback(infector_node, traceback_steps, hyper_edges, node_to_hyperedges_map)
            if ancestors:
                lineage_ancestor_sets.append(ancestors)
            else:
                return set()
        if not lineage_ancestor_sets: return set()
        final_candidates = lineage_ancestor_sets[0]
        for i in range(1, len(lineage_ancestor_sets)):
            final_candidates.intersection_update(lineage_ancestor_sets[i])
        return final_candidates
    return set()

def precompute_common_hyperedge_counts(hyper_edges):
    print("正在预处理节点对之间的连接强度...")
    common_counts = defaultdict(int)
    for he in hyper_edges:
        for i in range(len(he)):
            for j in range(i + 1, len(he)):
                pair = tuple(sorted((he[i], he[j])))
                common_counts[pair] += 1
    print(f"连接强度预处理完成，计算了 {len(common_counts)} 对节点的连接。")
    return common_counts


def _calculate_information_weighted_paths_v12(start_node, node_hyperdegrees, node_to_hyperedges_map,
                                              hyper_edges, common_hyperedge_counts):
    info_dist = defaultdict(lambda: float('inf'))
    hop_count = defaultdict(lambda: float('inf'))

    info_dist[start_node] = 0
    hop_count[start_node] = 0

    pq = [(0, start_node, 0)]

    while pq:
        dist, u, hops = heapq.heappop(pq)

        if dist > info_dist[u]:
            continue

        deg_u = node_hyperdegrees.get(u, 1)
        if deg_u == 0: continue  # 避免 log(0)

        for he_idx in node_to_hyperedges_map.get(u, set()):
            for v in hyper_edges[he_idx]:
                if v == u: continue

                common_uv = common_hyperedge_counts.get(tuple(sorted((u, v))), 0)
                if common_uv == 0: continue

                step_cost = np.log1p(deg_u) - np.log1p(common_uv)

                new_dist = dist + step_cost
                new_hops = hops + 1

                if new_dist < info_dist[v]:
                    info_dist[v] = new_dist
                    hop_count[v] = new_hops
                    heapq.heappush(pq, (new_dist, v, new_hops))

    return info_dist, hop_count

def precompute_all_pairs_costs(all_nodes, node_hyperdegrees, node_to_hyperedges_map, hyper_edges,
                               common_hyperedge_counts):
    all_pairs_info_dist = {}
    all_pairs_hop_count = {}

    start_time = time.time()
    for i, start_node in enumerate(all_nodes):
        info_dist, hop_count = _calculate_information_weighted_paths_v12(
            start_node, node_hyperdegrees, node_to_hyperedges_map, hyper_edges, common_hyperedge_counts
        )
        all_pairs_info_dist[start_node] = info_dist
        all_pairs_hop_count[start_node] = hop_count

    end_time = time.time()
    return all_pairs_info_dist, all_pairs_hop_count

def hypergraph_source_locator_v12(hyper_edges, node_to_hyperedges_map, sensor_inf,
                                  g_underlying, common_hyperedge_counts, all_nodes, path_info, dis_info,
                                  beta=0.5):
    max_paths = 10
    triggered_sensors = sorted(sensor_inf.items(), key=lambda item: item[1][0])
    valid_observations = [(s, (t, i)) for s, (t, i) in triggered_sensors if i != "unknown"]
    for i in sensor_inf:
        if sensor_inf[i][0] == 1:
            return [sensor_inf[i][1]], 100
    valid_observations = [(s, (t, i)) for s, (t, i) in triggered_sensors if i != "unknown"]
    if len(valid_observations) < 1:
        return [], 0  

    ref_observer, (ref_time, ref_infector) = valid_observations[0]
    try:
        dist_map_obs = nx.shortest_path_length(g_underlying, source=ref_observer)
        dist_map_inf = nx.shortest_path_length(g_underlying, source=ref_infector)
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return [], 0
    plausible_candidates = {s for s, d in dist_map_obs.items() if d == dist_map_inf.get(s, float('inf')) + 1}
    earliest_wave_sensors = [(s, info) for s, info in sensor_inf.items() if info[0] == ref_time]
    candidate_sources_v4 = generate_candidates_v4(earliest_wave_sensors, ref_time, hyper_edges, node_to_hyperedges_map
                                                  )
    candidate_sources = plausible_candidates.intersection(candidate_sources_v4)
    if not candidate_sources: candidate_sources = plausible_candidates if plausible_candidates else candidate_sources_v4
    if not candidate_sources: return [], 0
    num_candidates_initial = len(candidate_sources)

    node_hyperdegrees = {node: len(node_to_hyperedges_map.get(node, set())) for node in all_nodes}
    final_scores = defaultdict(float)

    for s in candidate_sources:
        total_penalty = 0.0
        is_valid_candidate = True
        path_p_total = 0
        distance_p_total = 0
        for sensor_node, (observed_time, infector_node) in valid_observations:
            target_node = infector_node

            try:
                pred_hops = all_pairs_hop_count.get(s, {}).get(target_node, float('inf'))
                min_info_cost = all_pairs_info_dist.get(s, {}).get(target_node, float('inf'))
                if min_info_cost == float('inf'):
                    is_valid_candidate = False
                    break

                distance_penalty = pred_hops
                info_path_penalty = min_info_cost
                weight = 1.0 / observed_time
                # combined_penalty = beta * distance_penalty + (1 - beta) * info_path_penalty
                distance_penalty += distance_penalty * weight
                path_p_total += info_path_penalty * weight

                # total_penalty += weight * combined_penalty

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                is_valid_candidate = False
                break

        if is_valid_candidate:
            # final_scores[s] = total_penalty
            final_scores[s] = beta * path_p_total + (1- beta) * distance_p_total

    if not final_scores:
        return list(candidate_sources), num_candidates_initial

    estimated_source = min(final_scores, key=final_scores.get)
    return [estimated_source], num_candidates_initial



if __name__ == "__main__":
    file_path_list = [
                      'data/Algebra.txt']
    for FILE_PATH in file_path_list:
        REPEAT_TIME = 1000
        SENSOR_RATIO = 0.1
        INFECTED_PROBABILITY = 0.5
        INFECTED_RATE = 0.1
        output_csv_file = 'pro_scale1/scale1.csv'
        csv_headers = [
        'Timestamp', 'Dataset', 'Repeat_Time', 'Sensor_Ratio',
        'Infection_Rate', 'Infection_Probability', 'Total_Time_Seconds',
        'Valid_Experiments', 'Correct_Predictions', 'Accuracy', 'Avg_Candidates',
        'Avg_Error_Distance' 
    ]

        if not os.path.exists(output_csv_file):
            with open(output_csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(csv_headers)

        hyper_edges = read_hypergraph_data(FILE_PATH)

        if hyper_edges is None: exit()

        all_nodes = sorted(list(set(node for he in hyper_edges for node in he)))

        node_to_hyperedges_map = defaultdict(set)
        for i, he in enumerate(hyper_edges):
            for node in he:
                node_to_hyperedges_map[node].add(i)

        common_hyperedge_counts = precompute_common_hyperedge_counts(hyper_edges)

        g_underlying = build_underlying_graph(hyper_edges, all_nodes)
        sensor_dir = "sensor_list_2"
        if not os.path.exists(sensor_dir):
            os.makedirs(sensor_dir)
        dataset_name = os.path.splitext(os.path.basename(FILE_PATH))[0]
        sensor_file_path = os.path.join(sensor_dir, f"sensors_TC_2_{dataset_name}_{SENSOR_RATIO}.txt")
        sensor_list = []
        if os.path.exists(sensor_file_path):
            with open(sensor_file_path, 'r') as f:
                sensor_ids_str = f.read()
                if sensor_ids_str:
                    sensor_list = [int(sid) for sid in sensor_ids_str.split(',')]
        else:
            num_vertices = len(all_nodes)
            node_to_hyperedge = {}
            hyperedge_to_nodes = {}
            for hyperedge_idx, hyperedge in enumerate(hyper_edges):
                for node in hyperedge:
                    if node in node_to_hyperedge:
                        node_to_hyperedge[node].append(hyperedge_idx)
                    else:
                        node_to_hyperedge[node] = [hyperedge_idx]
            for hyperedge_idx, hyperedge in enumerate(hyper_edges):
                hyperedge_to_nodes[hyperedge_idx] = hyperedge
            data_hyper = {}
            for hpe, nd_lst in hyperedge_to_nodes.items():
                data_hyper[str(hpe)] = set(nd_lst)

            sensor_num = int(num_vertices * SENSOR_RATIO)
            node_to_hyperedges_map = {node: set(h_edges) for node, h_edges in node_to_hyperedge.items()}
            hyperedge_to_nodes_map = {h_edge: set(nodes) for h_edge, nodes in hyperedge_to_nodes.items()}
            sensor_list = deploy_sensors_global_contribution(
                all_nodes,
                sensor_num,
                node_to_hyperedges_map,
                hyper_edges
            )
            os.makedirs(sensor_dir, exist_ok=True)
            with open(sensor_file_path, 'w') as f:
                f.write(','.join(map(str, sensor_list)))
        
        for INFECTED_RATE in np.arange(0.05, 0.31, 0.05):
            correct_predictions, valid_experiments, total_candidates_count = 0, 0, 0
            start_sim_time = time.time()
            real_time = 0

            precompute_dir = "path_penaty"
            if not os.path.exists(precompute_dir):
                os.makedirs(precompute_dir)
            dataset_name = os.path.splitext(os.path.basename(FILE_PATH))[0]
            precompute_filename = os.path.join(precompute_dir, f"{dataset_name}_costs.json")
            if os.path.exists(precompute_filename):
                node_hyperdegrees, all_pairs_info_dist, all_pairs_hop_count = load_precomputed_costs(precompute_filename)
                if node_hyperdegrees is None:
                    exit()
            else:
                node_hyperdegrees = {node: len(node_to_hyperedges_map.get(node, set())) for node in all_nodes}
                all_pairs_info_dist, all_pairs_hop_count = precompute_all_pairs_costs(
                    all_nodes, node_hyperdegrees, node_to_hyperedges_map, hyper_edges, common_hyperedge_counts
                )

                data_to_save = {
                    'node_hyperdegrees': node_hyperdegrees,
                    'all_pairs_info_dist': all_pairs_info_dist,
                    'all_pairs_hop_count': all_pairs_hop_count
                }
                save_precomputed_costs(data_to_save, precompute_filename)
            total_error_distance = 0.0
            while real_time < REPEAT_TIME:
                potential_sources = list(set(all_nodes) - set(sensor_list))
                if not potential_sources:
                    break
                actual_source = random.choice(potential_sources)

                infected_nodes = {actual_source}
                sensor_inf = {}
                current_infected_at_t = {actual_source}
                t = 0
                if actual_source in sensor_list:
                    sensor_inf[actual_source] = [0, "unknown"]

                while len(infected_nodes) / len(all_nodes) < INFECTED_RATE and t < 50:
                    t += 1
                    newly_infected_this_step = set()
                    nodes_to_spread_from = list(infected_nodes)
                    current_infected_at_t = set()
                    for infector_node in nodes_to_spread_from:
                        if infector_node in node_to_hyperedges_map:
                            he_idx = random.choice(list(node_to_hyperedges_map[infector_node]))
                            for target_node in hyper_edges[he_idx]:
                                if target_node not in infected_nodes and random.random() < INFECTED_PROBABILITY:
                                    newly_infected_this_step.add(target_node)
                                    current_infected_at_t.add(target_node)
                                    if target_node in sensor_list and target_node not in sensor_inf:
                                        sensor_inf[target_node] = [t, infector_node]
                    infected_nodes.update(newly_infected_this_step)

                if len(infected_nodes) / len(all_nodes) < INFECTED_RATE:
                    continue

                valid_experiments += 1
                real_time += 1

                predicted_source_list, candidate_count = hypergraph_source_locator_v12(
                    hyper_edges,
                    node_to_hyperedges_map,
                    sensor_inf,
                    g_underlying,
                    common_hyperedge_counts,
                    all_nodes,
                    all_pairs_info_dist,
                    all_pairs_hop_count,
                    beta=0.5  
                )
                total_candidates_count += candidate_count

                if predicted_source_list:
                    predicted_source = predicted_source_list[0]
                    if predicted_source == actual_source:
                        correct_predictions += 1
                    else:
                        try:
                            error_dist = all_pairs_hop_count.get(predicted_source, {}).get(actual_source, float('inf'))
                            if error_dist != float('inf'):
                                total_error_distance += error_dist
                        except KeyError:
                            pass

            end_sim_time = time.time()
            total_time = end_sim_time - start_sim_time
            accuracy, avg_candidates, avg_error_distance = 0.0, 0.0, 0.0


            if valid_experiments > 0:
                accuracy = correct_predictions / valid_experiments
                avg_candidates = total_candidates_count / valid_experiments
                print(f"最终准确率 (P_d): {accuracy:.4f} ({correct_predictions}/{valid_experiments})")
                print(f"平均候选集大小: {avg_candidates:.2f}")
            else:
                print("没有足够的数据来计算准确率。")

            if REPEAT_TIME > 0:
                avg_error_distance = total_error_distance / REPEAT_TIME
                print(f"平均错误距离 (AED): {avg_error_distance:.4f}")

            print("=" * 52)

            try:
                with open(output_csv_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    row_data = [
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        os.path.basename(FILE_PATH),
                        REPEAT_TIME,
                        SENSOR_RATIO,
                        INFECTED_RATE,
                        INFECTED_PROBABILITY,
                        f"{total_time:.2f}",
                        valid_experiments,
                        correct_predictions,
                        f"{accuracy:.4f}",
                        f"{avg_candidates:.2f}",
                        f"{avg_error_distance:.4f}"
                    ]
                    writer.writerow(row_data)
                print(f"本轮结果已成功追加到 {output_csv_file}")
            except Exception as e:
                print(f"错误：写入结果到CSV文件失败。原因: {e}")
