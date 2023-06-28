from flask import Blueprint, render_template
import folium
from datetime import datetime
import pandas as pd

views = Blueprint(__name__, "views")

@views.route("/")
def home():
    return render_template("index.html")

@views.route("/map")
def world_map():
    # Lecture des données des hôpitaux depuis le fichier CSV
    hospitals_data = pd.read_csv("preprocessed_data.csv")
    
    # Création de la carte folium centrée sur Paris
    map = folium.Map(location=[48.8566, 2.3522], zoom_start=12)

    # Ajout du marqueur pour la position actuelle à la Tour Eiffel
    folium.Marker(
        location=[48.8584, 2.2945],
        popup="Your position",
        icon=folium.Icon(color='red', icon='star', prefix='fa'),
        tooltip="Your position"
    ).add_to(map)

    # Ajout des marqueurs pour chaque hôpital
    for index, row in hospitals_data.iterrows():
        # Construction du contenu du popup avec une mise en forme personnalisée
        popup_content = f"<div style='font-size: 14px;'><strong>{row['name']}</strong></div>"
        popup_content += "<div style='font-size: 12px;'>"
        popup_content += f"Current waiting time (in min): <span style='color: {'green' if row['current waiting time'] < row['predicted waiting time in 3 hours'] else 'red'};'>{row['current waiting time']}</span><br>"
        popup_content += f"Predicted waiting time in 3 hours (in min): <span style='color: {'green' if row['predicted waiting time in 3 hours'] < row['current waiting time'] else 'red'};'>{row['predicted waiting time in 3 hours']}</span><br>"
        popup_content += f"Distance (in km): <span style='color: {'green' if row['distance'] < 20 else 'red'};'>{row['distance']}</span><br>"
        popup_content += f"Travel time (in min): <span style='color: {'green' if row['distance'] < 20 else 'red'};'>{row['travel time']}</span><br>"
        popup_content += f"Total time (in min): <span style='color: {'green' if row['total time'] < 120 else 'red'};'>{row['total time']}</span><br>"
        popup_content += "</div>"

        if int(row['total time']) < 60:
            marker_color = 'green'
        elif int(row['total time']) < 120:
            marker_color = 'orange'
        elif int(row['total time']) >= 120:
            marker_color = 'blue'

        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=popup_content,
            icon=folium.Icon(color=marker_color, icon='info-sign', prefix='fa'),
            tooltip=row['name']
        ).add_to(map)

    # Ajout de l'encart avec la date et l'heure actuelle en haut à gauche
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    map.get_root().html.add_child(folium.Element(f'<div id="timestamp" style="position: absolute; top: 10px; left: 10px; background-color: white; padding: 5px; font-weight: bold;">{timestamp}</div>'))

    # Personnalisation du style des marqueurs
    folium.TileLayer('cartodbpositron').add_to(map)
    folium.TileLayer('openstreetmap').add_to(map)
    folium.TileLayer('stamenterrain').add_to(map)
    folium.LayerControl().add_to(map)

    return map._repr_html_()
