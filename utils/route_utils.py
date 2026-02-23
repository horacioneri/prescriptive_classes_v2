import pandas as pd
import pydeck as pdk

def build_route_ids(raw_vars, problem):
    cities = problem["vars"]["cities"]
    name_to_id = {c["name"]: c["id"] for c in cities}
    ids = []
    if isinstance(raw_vars, dict) and "route" in raw_vars:
        candidate = raw_vars["route"]
    elif isinstance(raw_vars, dict):
        ordered = sorted(
            [(k, v) for k, v in raw_vars.items() if isinstance(v, (int, float))],
            key=lambda kv: kv[1],
        )
        candidate = [k for k, _ in ordered]
    elif isinstance(raw_vars, (list, tuple)):
        candidate = list(raw_vars)
    else:
        candidate = []

    for item in candidate:
        try:
            if isinstance(item, str):
                ids.append(name_to_id.get(item.strip(), int(item)))
            elif isinstance(item, dict) and "id" in item:
                ids.append(int(item["id"]))
            else:
                ids.append(int(item))
        except (ValueError, TypeError):
            continue

    # Keep only valid city ids
    ids = [i for i in ids if i in name_to_id.values()]

    # If nothing was parsed, fall back to the given city order so evaluation
    # never returns an empty route.
    if not ids:
        ids = [c["id"] for c in problem["vars"]["cities"]]

    # Ensure the tour is closed
    if ids and ids[0] != ids[-1]:
        ids.append(ids[0])

    return ids


def render_route_map(route_user, route_auto, problem, map_key):
    cities = problem["vars"]["cities"]
    coord = {c["id"]: {"lat": c["latitude"], "lon": c["longitude"], "name": c["name"]} for c in cities}

    def lines_from_route(route, color):
        if not route or len(route) < 2:
            return None
        records = []
        for i in range(len(route) - 1):
            s = coord[route[i]]
            t = coord[route[i + 1]]
            records.append(
                {"from_lon": s["lon"], "from_lat": s["lat"], "to_lon": t["lon"], "to_lat": t["lat"], "color": color}
            )
        return pd.DataFrame(records)

    user_df = lines_from_route(route_user, [30, 144, 255])
    auto_df = lines_from_route(route_auto, [255, 140, 0])
    point_df = pd.DataFrame(coord.values())

    layers = [
        pdk.Layer("ScatterplotLayer", point_df, get_position=["lon", "lat"], get_fill_color=[0, 0, 0], get_radius=35000),
        pdk.Layer("TextLayer", point_df, get_position=["lon", "lat"], get_text="name", get_size=12, get_color=[0, 0, 0]),
    ]
    if user_df is not None:
        layers.append(
            pdk.Layer(
                "LineLayer",
                user_df,
                get_source_position=["from_lon", "from_lat"],
                get_target_position=["to_lon", "to_lat"],
                get_width=4,
                get_color="color",
            )
        )
    if auto_df is not None:
        layers.append(
            pdk.Layer(
                "LineLayer",
                auto_df,
                get_source_position=["from_lon", "from_lat"],
                get_target_position=["to_lon", "to_lat"],
                get_width=4,
                get_color="color",
                get_dash_array=[4, 2],
            )
        )

    deck = pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=50, longitude=5, zoom=4),
        layers=layers,
        tooltip={"text": "{name}"},
    )
    return deck
