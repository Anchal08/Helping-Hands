from ffmpy import FFmpeg
SIGN_PATH = "/Users/user/Desktop/wchhack"
for real_name in ['0','1','2','4','6','8','9']:
    in_path = "/Users/user/Downloads/" +real_name + ".swf"
    out_path = SIGN_PATH + "/download/" + real_name + ".mp4"
    ff = FFmpeg(
    inputs = {in_path: None},
    outputs = {out_path: None})
    ff.run()
