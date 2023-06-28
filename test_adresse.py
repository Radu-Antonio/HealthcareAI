from geopy.geocoders import Nominatim
import pandas as pd
import random
hospitals_data = pd.read_csv("templates/france_hospitals_point.csv")

from math import radians, cos, sin, asin, acos, sqrt, pi
import numpy as np
def calculate_spherical_distance(lat1, lon1, lat2, lon2, r=6371):
    # Convert degrees to radians
    coordinates = lat1, lon1, lat2, lon2
    # radians(c) is same as c*pi/180
    phi1, lambda1, phi2, lambda2 = [
        radians(c) for c in coordinates
    ]  
    
    # Apply the haversine formula
    a = (np.square(sin((phi2-phi1)/2)) + cos(phi1) * cos(phi2) * 
         np.square(sin((lambda2-lambda1)/2)))
    d = 2*r*asin(np.sqrt(a))
    return d

your_position=[48.8584, 2.2945]

entete = ['name','latitude','longitude','speciality','emergency','current waiting time','predicted waiting time in 3 hours','distance','travel time','total time']
f = open('preprocessed_data'+'.csv','w')
ligneEntete = ",".join(entete) + "\n"
f.write(ligneEntete)
for index, row in hospitals_data.iterrows():
    address = row['alt_name']
    if type(address)==str:
        address_france=address+", France"
        # Créez une instance du géocodeur Nominatim
        geolocator = Nominatim(user_agent="hospital_locator")
        # Utilisez le géocodeur pour obtenir les coordonnées géographiques de l'adresse

        location = geolocator.geocode(address_france)
        if location:
            if str(row['emergency'])=='yes':
                latitude = location.latitude
                longitude = location.longitude
                distance=round(calculate_spherical_distance(latitude, longitude, your_position[0], your_position[1], r=6371),2)
                travel_time=int(2.5*distance**0.8)
                waiting_time=random.randint(30, 120)
                predicted_waiting_time=waiting_time+random.randint(10-waiting_time,45)
                total_time=min(waiting_time,predicted_waiting_time)+travel_time
                l=[address,str(latitude),str(longitude),str(row['healthcare-speciality']),str(row['emergency']),str(waiting_time),str(predicted_waiting_time), str(distance),str(travel_time),str(total_time)]
                line = ",".join(l) + "\n"
                f.write(line)
            else : 
                print('No urgency')
        else:
            print("unknown address")
f.close()

