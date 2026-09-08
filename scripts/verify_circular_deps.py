#!/usr/bin/env python3
"""
Circular dependency detection via Kahn's algorithm (efficient, per project).

For each project in dependencies.csv:
  - Build adjacency (predecessor -> successor) and in-degree counts
  - Run Kahn's topological sort
  - If sorted nodes < total nodes, a cycle exists
  - If cycle found, identify cycle member nodes via DFS on remaining subgraph

Kahn's algorithm runs in O(V+E) per project and avoids deep recursion.
Total: 100 projects, 18,176 edges — should be fast.
"""
import csv
import collections
import os

ROOT = "AI-Construction-Project-Risk-Delay-Predictor/data/raw/SCOPE_v02_Public"


def main():
    print("SCOPE v0.2 — Circular dependency check (Kahn's algorithm)")
    print("=" * 72)

    with open(os.path.join(ROOT, "dependencies.csv"), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        deps = list(reader)

    with open(os.path.join(ROOT, "activities.csv"), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        acts = list(reader)

    valid = {(a["project_id"], a["activity_id"]) for a in acts}

    by_project = collections.defaultdict(list)
    for d in deps:
        by_project[d["project_id"]].append(d)

    total_edges = len(deps)
    projects_checked = 0
    edges_checked = 0
    cycle_projects = []

    print(f"Dependencies total: {total_edges}")
    print(f"Projects with dependencies: {len(by_project)}")
    print()

    for pid in sorted(by_project.keys()):
        proj_deps = by_project[pid]
        projects_checked += 1
        edges_checked += len(proj_deps)

        nodes = set()
        adj = collections.defaultdict(list)
        in_deg = collections.defaultdict(int)

        for d in proj_deps:
            pred = d["predecessor_id"]
            succ = d["successor_id"]
            if (pid, pred) not in valid or (pid, succ) not in valid:
                continue
            nodes.add(pred)
            nodes.add(succ)
            adj[pred].append(succ)
            in_deg[succ] += 1
            if pred not in in_deg:
                in_deg[pred] = 0

        n_nodes = len(nodes)
        n_edges = sum(len(adj[n]) for n in nodes)

        queue = collections.deque(
            [n for n in nodes if in_deg.get(n, 0) == 0]
        )
        sorted_count = 0
        while queue:
            node = queue.popleft()
            sorted_count += 1
            for neighbor in adj[node]:
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:
                    queue.append(neighbor)

        if sorted_count < n_nodes:
            remaining = [n for n in nodes if in_deg.get(n, 0) > 0]
            sub_adj = collections.defaultdict(list)
            for n in remaining:
                for nb in adj[n]:
                    if nb in remaining:
                        sub_adj[n].append(nb)

            cycle_nodes = set()
            visited = set()
            stack = []

            def dfs(u):
                visited.add(u)
                stack.append(u)
                for v in sub_adj.get(u, []):
                    if v in stack:
                        idx = stack.index(v)
                        for node in stack[idx:]:
                            cycle_nodes.add(node)
                        return True
                    if v not in visited:
                        if dfs(v):
                            return True
                stack.pop()
                return False

            for n in remaining:
                if n not in visited:
                    dfs(n)

            entry = {
                "pid": pid,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "sorted_count": sorted_count,
                "remaining": len(remaining),
                "cycle_node_count": len(cycle_nodes),
                "cycle_nodes_sample": sorted(cycle_nodes)[:10],
            }
            cycle_projects.append(entry)

    print(f"Projects checked: {projects_checked}")
    print(f"Dependency edges checked: {edges_checked}")
    print(f"Projects with cycles: {len(cycle_projects)}")
    print()

    if cycle_projects:
        print("CYCLE DETAIL:")
        for cp in cycle_projects:
            print(
                f"  {cp['pid']}: {cp['n_nodes']} nodes, {cp['n_edges']} edges, "
                f"sorted={cp['sorted_count']}, "
                f"remaining(in cycle)={cp['remaining']}, "
                f"cycle_nodes={cp['cycle_node_count']}"
            )
            print(f"    sample cycle nodes: {cp['cycle_nodes_sample']}")
    else:
        print("NO CIRCULAR DEPENDENCIES FOUND in any project.")
        print("All 100 project dependency graphs are DAGs (directed acyclic graphs).")

    print()
    print("CONCLUSION:")
    if cycle_projects:
        print(
            f"  WARNING: {len(cycle_projects)} project(s) contain circular dependencies."
        )
        print("  These must be identified, documented, and handled before CPM (Phase 3).")
    else:
        print(
            "  All project dependency graphs are acyclic. No circular dependencies."
        )
        print(
            "  CPM can proceed on all 100 projects without cycle cleanup."
        )
    print()
    print("DONE")


if __name__ == "__main__":
    main()
