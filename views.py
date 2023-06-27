from flask import Blueprint, render_template, redirect, url_for
import folium

views = Blueprint(__name__, "views")

@views.route("/")
def home():
    return render_template("index.html")

@views.route("map")
def world_map():
    return folium.Map()._repr_html_()