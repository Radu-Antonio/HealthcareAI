from flask import Blueprint, render_template
import folium
import pandas as pd

views = Blueprint(__name__, "views")

@views.route("/")
def home():
    return render_template("index.html")

@views.route("/map")
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
        popup_content += "</div>"

        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=popup_content,
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(map)

    return map._repr_html_()

