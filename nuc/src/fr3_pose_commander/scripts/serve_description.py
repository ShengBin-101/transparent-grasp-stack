#!/usr/bin/env python3
"""Serve the robot description over HTTP so a remote Foxglove can draw the arm.

The URDF's meshes are package:// URIs, which mean nothing to a browser on the Orin (Foxglove
draws the frames and no arm). This rewrites each one to an http:// URL back here and serves
those packages' files. Point the Foxglove 3D panel's URDF layer at

    http://<this host>:8080/robot_description.urdf

The URDF comes from /robot_description, so it matches what franka_control loaded; start this
after the controller.
"""

import os
import re
import posixpath
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rospkg
import rospy

PACKAGE_URI = re.compile(rb"package://([A-Za-z0-9_]+)/")

CONTENT_TYPES = {
    ".urdf": "application/xml",
    ".xml": "application/xml",
    ".dae": "model/vnd.collada+xml",
    ".stl": "model/stl",
    ".obj": "text/plain",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


class Description(object):
    """The rewritten URDF and the package roots it is allowed to serve from."""

    def __init__(self, urdf, public_url):
        self.rospack = rospkg.RosPack()
        self.roots = {}
        self.urdf = PACKAGE_URI.sub(self._rewrite(public_url), urdf)

    def _rewrite(self, public_url):
        """re.sub callback: package://<name>/ -> <public_url>/pkg/<name>/, recording the root."""
        def replace(match):
            name = match.group(1).decode()
            try:
                self.roots[name] = os.path.realpath(self.rospack.get_path(name))
            except rospkg.ResourceNotFound:
                rospy.logwarn("serve_description: no package '%s'; its meshes "
                              "will not load in Foxglove", name)
                return match.group(0)
            return ("%s/pkg/%s/" % (public_url.rstrip("/"), name)).encode()
        return replace

    def resolve(self, url_path):
        """Map /pkg/<name>/<rel> to a real file, or None.

        Only returns paths inside a package the URDF refers to, so the open port cannot be
        used to read the rest of the disk.
        """
        parts = [p for p in url_path.split("/") if p not in ("", ".")]
        if len(parts) < 3 or parts[0] != "pkg":
            return None
        root = self.roots.get(parts[1])
        if root is None:
            return None
        candidate = os.path.realpath(os.path.join(root, *parts[2:]))
        if candidate != root and not candidate.startswith(root + os.sep):
            return None  # ../ escape
        return candidate if os.path.isfile(candidate) else None


class Handler(BaseHTTPRequestHandler):
    """Serves the rewritten URDF at / and /robot_description(.urdf), package files at /pkg/."""

    description = None  # set on the server before serving

    def do_GET(self):
        path = posixpath.normpath(self.path.split("?", 1)[0])

        if path in ("/robot_description.urdf", "/robot_description", "/"):
            return self._send(self.description.urdf, "application/xml")

        resolved = self.description.resolve(path)
        if resolved is None:
            self.send_error(404, "not served from here")
            return
        try:
            with open(resolved, "rb") as handle:
                body = handle.read()
        except OSError as exc:
            self.send_error(404, str(exc))
            return
        extension = os.path.splitext(resolved)[1].lower()
        self._send(body, CONTENT_TYPES.get(extension, "application/octet-stream"))

    def _send(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Foxglove's fetch is cross-origin.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        rospy.logdebug("serve_description: " + fmt, *args)


def _primary_address():
    """This machine's source address on the default route, from the routing table.

    gethostbyname(gethostname()) gives 127.0.1.1 on stock Ubuntu. Connecting a UDP socket
    sends no packets.
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        sock.close()


def main():
    rospy.init_node("serve_description")
    port = int(rospy.get_param("~port", 8080))
    # The host goes into the mesh URLs, so it must be an address the Orin can reach.
    host = rospy.get_param("~public_host", "").strip()
    if not host:
        host = (os.environ.get("ROS_IP") or os.environ.get("ROS_HOSTNAME") or "").strip()
    if not host or host.startswith("127."):
        host = _primary_address()
        rospy.logwarn("serve_description: no usable ROS_IP/ROS_HOSTNAME; "
                      "advertising meshes as %s. If Foxglove shows the frames "
                      "but no arm, set _public_host to this machine's address "
                      "on the Orin's network.", host)

    urdf = ""
    while not urdf and not rospy.is_shutdown():
        urdf = rospy.get_param("/robot_description", "")
        if not urdf:
            rospy.logwarn_throttle(
                5.0, "serve_description: waiting for /robot_description -- "
                     "start the controller first")
            rospy.sleep(1.0)
    if rospy.is_shutdown():
        return

    public_url = "http://%s:%d" % (host, port)
    Handler.description = Description(urdf.encode(), public_url)

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    rospy.loginfo("serve_description: URDF at %s/robot_description.urdf",
                  public_url)
    rospy.loginfo("serve_description: point the Foxglove 3D panel's URDF layer "
                  "at that URL (Add layer -> URDF -> URL)")
    rospy.loginfo("serve_description: serving meshes for: %s",
                  ", ".join(sorted(Handler.description.roots)) or "(none)")
    rospy.spin()
    server.shutdown()


if __name__ == "__main__":
    main()
