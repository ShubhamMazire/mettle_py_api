import threading
import time
import pymysql
import numpy as np
import os
import pandas as pd
from tensorflow import keras
from keras.models import model_from_json
from stl import mesh
from pyntcloud import PyntCloud
import stltovoxel
import trimesh
from werkzeug.utils import secure_filename
from datetime import datetime
import cadquery as cq
# from worker import worker
from mpl_toolkits import mplot3d
import requests


local="http://127.0.0.1:8000/api/customer/stlToPng"
prod = "https://metlle.com/api/customer/stlToPng"


api_url = local

def load_model_from_json_h5(json_path, h5_path):
    json_file = open(json_path, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(h5_path)
    return loaded_model


model_json = "pricing_model_withIRMR_sept21.json"
model_h5 = "pricing_model_withIRMR_sept21.h5"

loaded_model = load_model_from_json_h5(model_json, model_h5)

path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, "../uploads")


if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

THUMBNAIL_FOLDER = os.path.join(path, "../uploads/thumbnails")

if not os.path.isdir(THUMBNAIL_FOLDER):
    os.mkdir(THUMBNAIL_FOLDER)


ALLOWED_EXTENSIONS = set(["stl", "step"])


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS






def save_thumbnail(id):
    try:
        #    call api https://127.0.0.1:8000/customer/stlToPng in post method with id as form data
        #    response will be png file

        #    png = response.content
        
        data = {"id": id}

        try:
            response = requests.post(
                api_url, data=data, verify=False, timeout=0.00000001
            )
        except requests.exceptions.ReadTimeout:
            pass

    except Exception as e:
        pass
        # print("Error in creating thumbnail", e)
        return
    # with open(os.path.join(THUMBNAIL_FOLDER, filename), "wb") as f:
    #     f.write(png)

    # scene.
    # scene.show()

    # scene = trimesh.Scene()


def get_voxel(stl_file, id):
    vxl_files = []
    volumes = []
    print(f"Voxelizing1{stl_file}...........")
    my_mesh = mesh.Mesh.from_file(stl_file)

    save_thumbnail(str(id))

    voxel = np.array(stltovoxel.convert_file(stl_file))
    volume, cog, inertia = my_mesh.get_mass_properties()
    print(f"Voxelizing {voxel}...........")
    x_points = []
    y_points = []
    z_points = []

    for a in range(len(voxel)):
        x_points.append(voxel[a][0])
        y_points.append(voxel[a][1])
        z_points.append(voxel[a][2])

    vox_df = pd.DataFrame({"x": x_points, "y": y_points, "z": z_points})
    cloud = PyntCloud(vox_df)
    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=64, n_y=64, n_z=64)
    voxelgrid = cloud.structures[voxelgrid_id]
    binary_feature_vector = voxelgrid.get_feature_vector(mode="binary")
    Vox_1 = binary_feature_vector.reshape(64, 64, 64, 1)

    vxl_files.append(Vox_1)
    volumes.append(volume)

    Vox = Vox_1.reshape(1, 64, 64, 64, 1)
    return Vox


def get_mesh_from_step_file(file, id):
    # current_mesh = trimesh.Trimesh(**trimesh.interfaces.gmsh.load_gmsh(file_name=file, gmsh_args=[
    #     ("Mesh.Algorithm", 1),
    #     ("Mesh.CharacteristicLengthFromCurvature", 50),
    #     ("General.NumThreads", 10),
    #     ("Mesh.MinimumCirclePoints", 32)
    # ]))

    afile = cq.importers.importStep(file)

    print("-------------FIle name  ------", file)

    nm = file.split("/")[-1].split(".")[0] + ".stl"
    km = nm.replace("uploads\\", "")
    filename_stl = os.path.join(UPLOAD_FOLDER, km)
    cq.exporters.export(afile, filename_stl)

    # DELETE FILE FROM SERVER

    os.remove(file)

    # current_mesh.export(filename_stl)

    # update file path in database to column stl_file_path

    db = getConn()
    cursor = db.cursor()
    cursor.execute("UPDATE quote_requests SET file_path=%s WHERE id=%s", (km, id))
    db.commit()
    db.close()

    return filename_stl


def get_edge(stl_file):
    mesh1_e = trimesh.load(stl_file)
    mesh_edge = mesh1_e.edges_unique.shape[0]
    return np.array([mesh_edge])


def find_mins_maxs(obj):
    minx = obj.x.min()
    maxx = obj.x.max()
    miny = obj.y.min()
    maxy = obj.y.max()
    minz = obj.z.min()
    maxz = obj.z.max()
    return minx, maxx, miny, maxy, minz, maxz


def bounding_box(obj):
    minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(obj)
    l = maxx - minx
    b = maxy - miny
    w = maxz - minz
    return l, b, w


def bounding_box_vol(obj):
    l, b, w = bounding_box(obj)
    return l * b * w


def get_volume_and_irmr(input_file, id):
    if "step" in input_file.lower():
        stl_file = get_mesh_from_step_file(input_file, id)
    else:
        stl_file = input_file

    Mesh1 = mesh.Mesh.from_file(stl_file)
    volume, cog, inertia = Mesh1.get_mass_properties()
    bounding_box_volume = bounding_box_vol(Mesh1)
    mesh1_irmr = volume / bounding_box_volume
    boundingBox = bounding_box(Mesh1)
    area = Mesh1.areas.sum()

    return (
        np.array([volume]),
        np.array([mesh1_irmr]),
        stl_file,
        np.array([area]),
        boundingBox,
        bounding_box_volume,
    )


def predict_price(stl_file, id):
    data = {}

    vol, irmr, stl_file, area, boundingBox, bounding_box_volume = get_volume_and_irmr(
        stl_file, id
    )

    print(
        ":::::::::::::::::irmr, vol, vox, area, boundingBox",
        irmr,
        vol,
        area,
        boundingBox,
        bounding_box_volume,
    )

    data["volume"] = float(vol[0])
    data["irmr"] = float(irmr[0])
    data["area"] = float(area[0])
    data["boundingBox"] = str(list(boundingBox))
    data["bounding_box_volume"] = float(bounding_box_volume)


    print(float("{:.2f}".format(data["bounding_box_volume"])))

    # update above data in database

    db = getConn()
    cursor = db.cursor()
    cursor.execute(
        "UPDATE quote_requests SET volume=%s,irmr=%s,surface_area=%s,bounding_box=%s,bounding_box_volume=%s WHERE id=%s",
        (
            float("{:.2f}".format(data["volume"])),
            float("{:.2f}".format(data["irmr"])),
            float("{:.2f}".format(data["area"])),
            data["boundingBox"],
            float("{:.2f}".format(data["bounding_box_volume"])),
            id,
        ),
    )
    db.commit()
    db.close()

    vox = get_voxel(stl_file, id)  # this takes time
    # edge = get_edge(stl_file)
    x = [irmr, vol, vox]

    y_pred = loaded_model.predict(x)

    print(":::::::::::::::::::price(y_pred)", y_pred)

    return {
        "price": float(y_pred),
        "volume": float(vol[0]),
        "IRMR": float(irmr[0]),
        "area": float(area[0]),
        "boundingBox": str(list(boundingBox)),
        "bounding_box_volume": float(bounding_box_volume),
    }


# Function to process 3D file and update results in the database
def process_3d_file(entry):
    print("Processing 3D file..." + str(entry["id"]) + "-" + entry["file_path"])
    filepath = entry["file_path"]

    absFilePath = os.path.join(UPLOAD_FOLDER, filepath)
    print("absFilePath", absFilePath)

    data = predict_price(absFilePath, entry["id"])

    print("Processing complete" + str(entry["id"]) + "-" + entry["file_path"])
    

    # Example data to be updated in the database
    irmr = float("{:.2f}".format(data["IRMR"]))
    surface_area = float("{:.2f}".format(data["area"]))
    bounding_box = data["boundingBox"]
    predicted_cost = float("{:.2f}".format(data["price"]))
    volume = float("{:.2f}".format(data["volume"]))
    bounding_box_volume = float("{:.2f}".format(data["bounding_box_volume"]))
    print('data1')
    # Update the database with the results
    update_query = (
        f"UPDATE quote_requests SET status='finalized', "
        f"irmr={irmr}, surface_area={surface_area}, "
        f"bounding_box='{bounding_box}', predicted_cost={predicted_cost}, "
        f"bounding_box_volume={bounding_box_volume},"
        f"volume={volume}, quote_type='auto_generated' WHERE id={entry['id']}"
    )

    db = pymysql.connect(
        host="52.66.107.82",
        port=3306,
        user="vivek-web",
        password="13121312",
        database="metlledb",
    )
    print('data')

    # db = pymysql.connect(
    #     host="localhost",
    #     port=3306,
    #     user="root",
    #     password="",
    #     database="metlle-db",
    # )
    print(data)

    with db.cursor() as cursor:
        cursor.execute(update_query)
    db.commit()


# Function to periodically check the database for new entries and process them


def getConn():
    # pymysql.connect(
    #     host="217.21.84.154",
    #     port=3306,
    #     user="u946054600_metlle",
    #     password="MiM9N*~[|7",
    #     database="u946054600_metlle",
    # )

    return pymysql.connect(
        host="localhost",
        port=3306,
        user="root",
        password="",
        database="metlle-db",
    )


def check_database():
    while True:
        try:
            db = getConn()
        except Exception as e:
            print("Error in database connection")
            time.sleep(30)
            continue

        # Fetch pending entries from the database
        select_query = "SELECT * FROM quote_requests WHERE status='pending'"
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute(select_query)
        data = cursor.fetchall()
        db.close()

        for row in data:
            print('Update the status to processing:'+str(row['id']))
            update_status_query = (
                f"UPDATE quote_requests SET status='processing' WHERE id={row['id']}"
            )

            db = getConn()

            with db.cursor() as cursor:
                cursor.execute(update_status_query)
            db.commit()
            db.close()

            # Create a thread to process the 3D file
            thread = threading.Thread(target=process_3d_file, args=(row,))
            # thread.daemon = True
            thread.start()
            print(
                "Thread started"
                + str(row["id"])
                + "-"
                + row["file_path"]
                + " Time:"
                + str(datetime.now())
            )
            # close the thread after 1 minute
            thread.join(60)

            if thread.is_alive():
                print(
                    "Thread alive"
                    + str(row["id"])
                    + "-"
                    + row["file_path"]
                    + " Time:"
                    + str(datetime.now())
                )
                # Update the status to 'pending' if the thread is still alive
                update_status_query = f"UPDATE quote_requests SET quote_type='manual_generated', status='manual' WHERE id={row['id']}"
                db = getConn()
                with db.cursor() as cursor:
                    cursor.execute(update_status_query)
                db.commit()
                db.close()

                print(
                    "Thread closed"
                    + str(row["id"])
                    + "-"
                    + row["file_path"]
                    + " Time:"
                    + str(datetime.now())
                )
                # thread._stop()
        print("30 seconds sleep")
        time.sleep(30)  # Wait for 30 seconds before checking again


def start_check_db():
    check_thread = threading.Thread(target=check_database)
    check_thread.start()
    print("Thread started")
    check_thread.join()


if __name__ == "__main__":
    while True:
        try:
            start_check_db()
        except Exception as e:
            print("Error in main thread")
        time.sleep(30)
