from flask import Blueprint, render_template
import folium
import pandas as pd
from math import sin, cos, atan2, pi

views = Blueprint(__name__, "views")


def get_time(user_position, hospital_position):
    lat1, long1 = user_position
    lat2, long2 = hospital_position
    R = 6371e3
    phi1 = lat1 * pi / 180
    phi2 = lat2 * pi / 180
    delta_phi = (lat2 - lat1) * pi / 180
    delta_lambda = (long2 - long1) * pi / 180

    a = sin(delta_phi / 2) ** 2 + cos(phi1) * \
        cos(phi2) * sin(delta_lambda / 2) ** 2
    c = 2 * atan2(a ** 0.5, (1 - a) ** 0.5)
    d = R * c  # distance in meters
    if d < 5000:
        velocity = 536.448
    velocity = 1341.12  # meters / mins
    return d / velocity

print(get_time((48.8584, 2.2945), (48.85468295, 2.348861910166425)))

@views.route("/")
def world_map():
    # Lecture des données des hôpitaux depuis le fichier CSV
    hospitals_data = pd.read_csv("preprocessed_data.csv")

    # Création de la carte folium centrée sur la France
    map = folium.Map(location=[48.8566, 2.3522], zoom_start=12)

    # Ajout du marqueur pour la position actuelle à la Tour Eiffel
    folium.Marker(
        location=[48.8584, 2.2945],
        popup="Your position",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(map)

    # Ajout des marqueurs pour chaque hôpital
    for index, row in hospitals_data.iterrows():
        # Construction du contenu du popup avec une mise en forme personnalisée
        popup_content = f"<div style='font-size: 14px;'><strong>{row['name']}</strong></div>"
        popup_content += "<div style='font-size: 12px;'>"
        popup_content += f"Current wainting time: <span style='color: {'green' if row['current waiting time'] < row['predicted waiting time in 3 hours'] else 'red'};'>{row['current waiting time']}</span><br>"
        popup_content += f"Predicted waiting time in 3 hours: <span style='color: {'green' if row['predicted waiting time in 3 hours'] < row['current waiting time'] else 'red'};'>{row['predicted waiting time in 3 hours']}</span>"
        popup_content += f"Distance: <span style='color: {'green'}'> {get_time((48.8584, 2.2945), (row['latitude'], row['longitude']))}</span>"
         # if row['predicted waiting time in 3 hours'] < row['current waiting time'] else 'red'};'>{row['predicted waiting time in 3 hours
        popup_content += "</div>"

        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=popup_content,
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(map)

    return map._repr_html_()
