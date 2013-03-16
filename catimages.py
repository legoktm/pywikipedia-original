#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Image by content categorization derived from 'checkimages.py'.

Script to check uncategorized files. This script checks if a file
has some content that allows to assign it to a category.

This script runs on commons only. It needs also external libraries
(see imports and comments there) and additional configuration/data
files in order to run properly. Most of them can be checked-out at:
    http://svn.toolserver.org/svnroot/drtrigon/
(some code might get compiled on-the-fly, so a GNU compiler along
with library header files is needed too)

This script understands the following command-line arguments:

-cat[:#]            Use a category as recursive generator
                    (if no given 'Category:Media_needing_categories' is used)

-start[:#]          Start after File:[:#] or if no file given start from top
                    (instead of resuming last run).

-limit              The number of images to check (default: 80)

-noguesses          If given, this option will disable all guesses (which are
                    less reliable than true searches).

-single:#           Run for one (any) single page only.

-train              Train classifiers on good (homegenous) categories.

X-sendemail          Send an email after tagging.

X-untagged[:#]       Use daniel's tool as generator:
X                    http://toolserver.org/~daniel/WikiSense/UntaggedImages.php


"""

#
# (C) Kyle/Orgullomoore, 2006-2007 (newimage.py)
# (C) Pywikipedia team, 2007-2011 (checkimages.py)
# (C) DrTrigon, 2012
#
# Distributed under the terms of the MIT license.
#
__version__ = '$Id$'
#

# python default packages
import re, urllib2, os, locale, sys, datetime, math, shutil, mimetypes, shelve
import StringIO, json # fallback: simplejson
from subprocess import Popen, PIPE
import Image
#import ImageFilter

scriptdir = os.path.dirname(sys.argv[0])
if not os.path.isabs(scriptdir):
    scriptdir = os.path.abspath(os.path.join(os.curdir, scriptdir))

# additional python packages (more exotic and problematic ones)
try:
    import numpy as np
    from scipy import ndimage, fftpack, linalg
    import cv
    # TS: nonofficial cv2.so backport of the testing-version of
    # python-opencv because of missing build-host, done by DaB
    sys.path.append('/usr/local/lib/python2.6/')
    import cv2
    sys.path.remove('/usr/local/lib/python2.6/')
    import pyexiv2
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import gtk                  # ignore warning: "GtkWarning: could not open display"
    import rsvg                     # gnome-python2-rsvg (binding to librsvg)
    import cairo
    import magic                    # python-magic (binding to libmagic)
except:
    # either raise the ImportError later or skip it
    pass
# modules needing compilation are imported later on request:
# (see https://jira.toolserver.org/browse/TS-1452)
# e.g. opencv, jseg, slic, pydmtx, zbar, (pyml or equivalent)
# binaries: exiftool, pdftotext/pdfimages (poppler), ffprobe (ffmpeg),
#           convert/identify (ImageMagick), (ocropus)
# TODO:
#   (pdfminer not used anymore/at the moment...)
#   python-djvulibre or python-djvu for djvu support

# pywikipedia framework python packages
import wikipedia as pywikibot
import pagegenerators, catlib
import checkimages

# DrTrigonBot framework packages
import dtbext.pycolorname as pycolorname
#import dtbext._mlpy as mlpy
target = os.path.join(scriptdir, 'dtbext')
sys.path.append(target)
from colormath.color_objects import RGBColor
from py_w3c.validators.html.validator import HTMLValidator, ValidationFault
sys.path.remove(target)
#from dtbext.pdfminer import pdfparser, pdfinterp, pdfdevice, converter, cmapdb, layout

locale.setlocale(locale.LC_ALL, '')


###############################################################################
# <--------------------------- Change only below! --------------------------->#
###############################################################################

# NOTE: in the messages used by the Bot if you put __botnick__ in the text, it
# will automatically replaced with the bot's nickname.

# Add your project (in alphabetical order) if you want that the bot start
project_inserted = [u'commons',]

# Ok, that's all. What is below, is the rest of code, now the code is fixed and it will run correctly in your project.
################################################################################
# <--------------------------- Change only above! ---------------------------> #
################################################################################

tmpl_FileContentsByBot = u"""}}
{{FileContentsByBot
| botName = ~~~
|"""

# this list is auto-generated during bot run (may be add notifcation about NEW templates)
#tmpl_available_spec = [ u'Properties', u'ColorRegions', u'Faces', u'ColorAverage' ]
tmpl_available_spec = []    # auto-generated


# global
useGuesses = True        # Use guesses which are less reliable than true searches


# all detection and recognition methods - bindings to other classes, modules and libs
class FileData(object):
    # .../opencv/samples/c/facedetect.cpp
    # http://opencv.willowgarage.com/documentation/python/genindex.html
    def _detect_Faces_CV(self):
        """Converts an image to grayscale and prints the locations of any
           faces found"""
        # http://python.pastebin.com/m76db1d6b
        # http://creatingwithcode.com/howto/face-detection-in-static-images-with-python/
        # http://opencv.willowgarage.com/documentation/python/objdetect_cascade_classification.html
        # http://opencv.willowgarage.com/wiki/FaceDetection
        # http://blog.jozilla.net/2008/06/27/fun-with-python-opencv-and-face-detection/
        # http://www.cognotics.com/opencv/servo_2007_series/part_4/index.html

        # skip file formats not supported (yet?)
        if (self.image_mime[1] in ['ogg', 'pdf', 'vnd.djvu']):
            return

        # https://code.ros.org/trac/opencv/browser/trunk/opencv_extra/testdata/gpu/haarcascade?rev=HEAD
        xml = os.path.join(scriptdir, 'dtbext/opencv/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
        #xml = os.path.join(scriptdir, 'dtbext/opencv/haarcascades/haarcascade_eye.xml')
        if not os.path.exists(xml):
            raise IOError(u"No such file: '%s'" % xml)
        #nestedCascade = cv.Load(
        nestedCascade = cv2.CascadeClassifier(xml)
        # http://tutorial-haartraining.googlecode.com/svn/trunk/data/haarcascades/
        xml = os.path.join(scriptdir, 'dtbext/opencv/haarcascades/haarcascade_frontalface_alt.xml')
        # MAY BE USE 'haarcascade_frontalface_alt_tree.xml' ALSO / INSTEAD...?!!
        if not os.path.exists(xml):
            raise IOError(u"No such file: '%s'" % xml)
        #cascade       = cv.Load(
        cascade       = cv2.CascadeClassifier(xml)
        xml = os.path.join(scriptdir, 'dtbext/opencv/haarcascades/haarcascade_profileface.xml')
        if not os.path.exists(xml):
            raise IOError(u"No such file: '%s'" % xml)
        cascadeprofil = cv2.CascadeClassifier(xml)
        xml = os.path.join(scriptdir, 'dtbext/opencv/haarcascades/haarcascade_mcs_mouth.xml')
        if not os.path.exists(xml):
            raise IOError(u"No such file: '%s'" % xml)
        cascademouth = cv2.CascadeClassifier(xml)
        xml = os.path.join(scriptdir, 'dtbext/opencv/haarcascades/haarcascade_mcs_nose.xml')
        if not os.path.exists(xml):
            raise IOError(u"No such file: '%s'" % xml)
        cascadenose = cv2.CascadeClassifier(xml)
        xml = os.path.join(scriptdir, 'dtbext/opencv/haarcascades/haarcascade_lefteye_2splits.xml')
        if not os.path.exists(xml):
            raise IOError(u"No such file: '%s'" % xml)
        cascadelefteye = cv2.CascadeClassifier(xml)        # (http://yushiqi.cn/research/eyedetection)
        xml = os.path.join(scriptdir, 'dtbext/opencv/haarcascades/haarcascade_righteye_2splits.xml')
        if not os.path.exists(xml):
            raise IOError(u"No such file: '%s'" % xml)
        cascaderighteye = cv2.CascadeClassifier(xml)       # (http://yushiqi.cn/research/eyedetection)
        xml = os.path.join(scriptdir, 'dtbext/opencv/haarcascades/haarcascade_mcs_leftear.xml')
        if not os.path.exists(xml):
            raise IOError(u"No such file: '%s'" % xml)
        cascadeleftear = cv2.CascadeClassifier(xml)
        xml = os.path.join(scriptdir, 'dtbext/opencv/haarcascades/haarcascade_mcs_rightear.xml')
        if not os.path.exists(xml):
            raise IOError(u"No such file: '%s'" % xml)
        cascaderightear = cv2.CascadeClassifier(xml)

        #self._info['Faces'] = []
        scale = 1.
        # So, to find an object of an unknown size in the image the scan
        # procedure should be done several times at different scales.
        # http://opencv.itseez.com/modules/objdetect/doc/cascade_classification.html
        try:
            #image = cv.LoadImage(self.image_path)
            #img    = cv2.imread( self.image_path, 1 )
            img    = cv2.imread( self.image_path_JPEG, 1 )
            #image  = cv.fromarray(img)
            if img == None:
                raise IOError
            
            # !!! the 'scale' here IS RELEVANT FOR THE DETECTION RATE;
            # how small and how many features are detected as faces (or eyes)
            scale  = max([1., np.average(np.array(img.shape)[0:2]/500.)])
        except IOError:
            pywikibot.output(u'WARNING: unknown file type [_detect_Faces_CV]')
            return
        except AttributeError:
            pywikibot.output(u'WARNING: unknown file type [_detect_Faces_CV]')
            return

        #detectAndDraw( image, cascade, nestedCascade, scale );
        # http://nullege.com/codes/search/cv.CvtColor
        #smallImg = cv.CreateImage( (cv.Round(img.shape[0]/scale), cv.Round(img.shape[1]/scale)), cv.CV_8UC1 )
        #smallImg = cv.fromarray(np.empty( (cv.Round(img.shape[0]/scale), cv.Round(img.shape[1]/scale)), dtype=np.uint8 ))
        smallImg = np.empty( (cv.Round(img.shape[1]/scale), cv.Round(img.shape[0]/scale)), dtype=np.uint8 )

        #cv.CvtColor( image, gray, cv.CV_BGR2GRAY )
        gray = cv2.cvtColor( img, cv.CV_BGR2GRAY )
        #cv.Resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR )        
        smallImg = cv2.resize( gray, smallImg.shape, interpolation=cv2.INTER_LINEAR )
        #cv.EqualizeHist( smallImg, smallImg )
        smallImg = cv2.equalizeHist( smallImg )

        t = cv.GetTickCount()
        faces = list(cascade.detectMultiScale( smallImg,
            1.1, 2, 0
            #|cv.CV_HAAR_FIND_BIGGEST_OBJECT
            #|cv.CV_HAAR_DO_ROUGH_SEARCH
            |cv.CV_HAAR_SCALE_IMAGE,
            (30, 30) ))
        #faces = cv.HaarDetectObjects(grayscale, cascade, storage, 1.2, 2,
        #                           cv.CV_HAAR_DO_CANNY_PRUNING, (50,50))
        facesprofil = list(cascadeprofil.detectMultiScale( smallImg,
            1.1, 2, 0
            #|cv.CV_HAAR_FIND_BIGGEST_OBJECT
            #|cv.CV_HAAR_DO_ROUGH_SEARCH
            |cv.CV_HAAR_SCALE_IMAGE,
            (30, 30) ))
        #faces = self._util_merge_Regions(faces + facesprofil)[0]
        faces = self._util_merge_Regions(faces + facesprofil, overlap=True)[0]
        faces = np.array(faces)
        #if faces:
        #    self._drawRect(faces) #call to a python pil
        t = cv.GetTickCount() - t
        #print( "detection time = %g ms\n" % (t/(cv.GetTickFrequency()*1000.)) )
        #colors = [ (0,0,255),
        #    (0,128,255),
        #    (0,255,255),
        #    (0,255,0),
        #    (255,128,0),
        #    (255,255,0),
        #    (255,0,0),
        #    (255,0,255) ]
        result = []
        for i, r in enumerate(faces):
            #color = colors[i%8]
            (rx, ry, rwidth, rheight) = r
            #cx = cv.Round((rx + rwidth*0.5)*scale)
            #cy = cv.Round((ry + rheight*0.5)*scale)
            #radius = cv.Round((rwidth + rheight)*0.25*scale)
            #cv2.circle( img, (cx, cy), radius, color, 3, 8, 0 )
            #if nestedCascade.empty():
            #    continue
            # Wilson, Fernandez: FACIAL FEATURE DETECTION USING HAAR CLASSIFIERS
            # http://nichol.as/papers/Wilson/Facial%20feature%20detection%20using%20Haar.pdf
            #dx, dy = cv.Round(rwidth*0.5), cv.Round(rheight*0.5)
            dx, dy = cv.Round(rwidth/8.), cv.Round(rheight/8.)
            (rx, ry, rwidth, rheight) = (max([rx-dx,0]), max([ry-dy,0]), min([rwidth+2*dx,img.shape[1]]), min([rheight+2*dy,img.shape[0]]))
            #smallImgROI = smallImg
            #print r, (rx, ry, rwidth, rheight)
            #smallImgROI = smallImg[ry:(ry+rheight),rx:(rx+rwidth)]
            smallImgROI = smallImg[ry:(ry+6*dy),rx:(rx+rwidth)] # speed up by setting instead of extracting ROI
            nestedObjects = nestedCascade.detectMultiScale( smallImgROI,
                1.1, 2, 0
                #|CV_HAAR_FIND_BIGGEST_OBJECT
                #|CV_HAAR_DO_ROUGH_SEARCH
                #|CV_HAAR_DO_CANNY_PRUNING
                |cv.CV_HAAR_SCALE_IMAGE,
                (30, 30) )
            nestedObjects = self._util_merge_Regions(list(nestedObjects), overlap=True)[0]
            if len(nestedObjects) < 2:
                nestedLeftEye = cascadelefteye.detectMultiScale( smallImgROI,
                    1.1, 2, 0
                    #|CV_HAAR_FIND_BIGGEST_OBJECT
                    #|CV_HAAR_DO_ROUGH_SEARCH
                    #|CV_HAAR_DO_CANNY_PRUNING
                    |cv.CV_HAAR_SCALE_IMAGE,
                    (30, 30) )
                nestedRightEye = cascaderighteye.detectMultiScale( smallImgROI,
                    1.1, 2, 0
                    #|CV_HAAR_FIND_BIGGEST_OBJECT
                    #|CV_HAAR_DO_ROUGH_SEARCH
                    #|CV_HAAR_DO_CANNY_PRUNING
                    |cv.CV_HAAR_SCALE_IMAGE,
                    (30, 30) )
                nestedObjects = self._util_merge_Regions(list(nestedObjects) +
                                                  list(nestedLeftEye) + 
                                                  list(nestedRightEye), overlap=True)[0]
            #if len(nestedObjects) > 2:
            #    nestedObjects = self._util_merge_Regions(list(nestedObjects), close=True)[0]
            smallImgROI = smallImg[(ry+4*dy):(ry+rheight),rx:(rx+rwidth)]
            nestedMouth = cascademouth.detectMultiScale( smallImgROI,
                1.1, 2, 0
                |cv.CV_HAAR_FIND_BIGGEST_OBJECT
                |cv.CV_HAAR_DO_ROUGH_SEARCH
                #|CV_HAAR_DO_CANNY_PRUNING
                |cv.CV_HAAR_SCALE_IMAGE,
                (30, 30) )
            smallImgROI = smallImg[(ry+(5*dy)/2):(ry+5*dy+(5*dy)/2),(rx+(5*dx)/2):(rx+5*dx+(5*dx)/2)]
            nestedNose = cascadenose.detectMultiScale( smallImgROI,
                1.1, 2, 0
                |cv.CV_HAAR_FIND_BIGGEST_OBJECT
                |cv.CV_HAAR_DO_ROUGH_SEARCH
                #|CV_HAAR_DO_CANNY_PRUNING
                |cv.CV_HAAR_SCALE_IMAGE,
                (30, 30) )
            smallImgROI = smallImg[(ry+2*dy):(ry+6*dy),rx:(rx+rwidth)]
            nestedEars = list(cascadeleftear.detectMultiScale( smallImgROI,
                1.1, 2, 0
                |cv.CV_HAAR_FIND_BIGGEST_OBJECT
                |cv.CV_HAAR_DO_ROUGH_SEARCH
                #|CV_HAAR_DO_CANNY_PRUNING
                |cv.CV_HAAR_SCALE_IMAGE,
                (30, 30) ))
            nestedEars += list(cascaderightear.detectMultiScale( smallImgROI,
                1.1, 2, 0
                |cv.CV_HAAR_FIND_BIGGEST_OBJECT
                |cv.CV_HAAR_DO_ROUGH_SEARCH
                #|CV_HAAR_DO_CANNY_PRUNING
                |cv.CV_HAAR_SCALE_IMAGE,
                (30, 30) ))
            data = { 'ID':       (i+1),
                     'Position': tuple(np.int_(r*scale)), 
                     'Type':     u'-',
                     'Eyes':     [],
                     'Mouth':    (),
                     'Nose':     (),
                     'Ears':     [], }
            data['Coverage'] = float(data['Position'][2]*data['Position'][3])/(self.image_size[0]*self.image_size[1])
            #if (c >= confidence):
            #    eyes = nestedObjects
            #    if not (type(eyes) == type(tuple())):
            #        eyes = tuple((eyes*scale).tolist())
            #    result.append( {'Position': r*scale, 'eyes': eyes, 'confidence': c} )
            #print {'Position': r, 'eyes': nestedObjects, 'confidence': c}
            for nr in nestedObjects:
                (nrx, nry, nrwidth, nrheight) = nr
                cx = cv.Round((rx + nrx + nrwidth*0.5)*scale)
                cy = cv.Round((ry + nry + nrheight*0.5)*scale)
                radius = cv.Round((nrwidth + nrheight)*0.25*scale)
                #cv2.circle( img, (cx, cy), radius, color, 3, 8, 0 )
                data['Eyes'].append( (cx-radius, cy-radius, 2*radius, 2*radius) )
            if len(nestedMouth):
                (nrx, nry, nrwidth, nrheight) = nestedMouth[0]
                cx = cv.Round((rx + nrx + nrwidth*0.5)*scale)
                cy = cv.Round(((ry+4*dy) + nry + nrheight*0.5)*scale)
                radius = cv.Round((nrwidth + nrheight)*0.25*scale)
                #cv2.circle( img, (cx, cy), radius, color, 3, 8, 0 )
                data['Mouth'] = (cx-radius, cy-radius, 2*radius, 2*radius)
            if len(nestedNose):
                (nrx, nry, nrwidth, nrheight) = nestedNose[0]
                cx = cv.Round(((rx+(5*dx)/2) + nrx + nrwidth*0.5)*scale)
                cy = cv.Round(((ry+(5*dy)/2) + nry + nrheight*0.5)*scale)
                radius = cv.Round((nrwidth + nrheight)*0.25*scale)
                #cv2.circle( img, (cx, cy), radius, color, 3, 8, 0 )
                data['Nose'] = (cx-radius, cy-radius, 2*radius, 2*radius)
            for nr in nestedEars:
                (nrx, nry, nrwidth, nrheight) = nr
                cx = cv.Round((rx + nrx + nrwidth*0.5)*scale)
                cy = cv.Round((ry + nry + nrheight*0.5)*scale)
                radius = cv.Round((nrwidth + nrheight)*0.25*scale)
                #cv2.circle( img, (cx, cy), radius, color, 3, 8, 0 )
                data['Ears'].append( (cx-radius, cy-radius, 2*radius, 2*radius) )
            result.append( data )

        ## see '_drawRect'
        #if result:
        #    #image_path_new = os.path.join(scriptdir, 'cache/0_DETECTED_' + self.image_filename)
        #    image_path_new = self.image_path_JPEG.replace(u"cache/", u"cache/0_DETECTED_")
        #    cv2.imwrite( image_path_new, img )

        #return faces.tolist()
        self._info['Faces'] += result
        return

    # .../opencv/samples/cpp/peopledetect.cpp
    # + Haar/Cascade detection
    def _detect_People_CV(self):
        # http://stackoverflow.com/questions/10231380/graphic-recognition-of-people
        # https://code.ros.org/trac/opencv/ticket/1298
        # http://opencv.itseez.com/modules/gpu/doc/object_detection.html
        # http://opencv.willowgarage.com/documentation/cpp/basic_structures.html
        # http://www.pygtk.org/docs/pygtk/class-gdkrectangle.html
        
        self._info['People'] = []
        scale = 1.
        try:
            img = cv2.imread(self.image_path_JPEG, 1)

            if (img == None) or (min(img.shape[:2]) < 100) or (not img.data) \
               or (self.image_size[0] is None):
                return

            # !!! the 'scale' here IS RELEVANT FOR THE DETECTION RATE;
            # how small and how many features are detected
            #scale  = max([1., np.average(np.array(img.shape)[0:2]/500.)])
            scale  = max([1., np.average(np.array(img.shape)[0:2]/400.)])
            #scale  = max([1., np.average(np.array(img.shape)[0:2]/300.)])
        except IOError:
            pywikibot.output(u'WARNING: unknown file type [_detect_People_CV]')
            return
        except AttributeError:
            pywikibot.output(u'WARNING: unknown file type [_detect_People_CV]')
            return

        # similar to face detection
        smallImg = np.empty( (cv.Round(img.shape[1]/scale), cv.Round(img.shape[0]/scale)), dtype=np.uint8 )
        #gray = cv2.cvtColor( img, cv.CV_BGR2GRAY )
        gray = img
        smallImg = cv2.resize( gray, smallImg.shape, interpolation=cv2.INTER_LINEAR )
        #smallImg = cv2.equalizeHist( smallImg )
        img = smallImg
        
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        #cv2.namedWindow("people detector", 1)
        
        found = found_filtered = []
        #t = time.time()
        # run the detector with default parameters. to get a higher hit-rate
        # (and more false alarms, respectively), decrease the hitThreshold and
        # groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
        # detectMultiScale(img, hit_threshold=0, win_stride=Size(),
        #                  padding=Size(), scale0=1.05, group_threshold=2)
        enable_recovery()   # enable recovery from hard crash
        found = list(hog.detectMultiScale(img, 0.25, (8,8), (32,32), 1.05, 2))
        disable_recovery()  # disable since everything worked out fine

        # people haar/cascaded classifier
        # use 'haarcascade_fullbody.xml', ... also (like face detection)
        xml = os.path.join(scriptdir, 'dtbext/opencv/haarcascades/haarcascade_fullbody.xml')
        #xml = os.path.join(scriptdir, 'dtbext/opencv/haarcascades/haarcascade_lowerbody.xml')
        #xml = os.path.join(scriptdir, 'dtbext/opencv/haarcascades/haarcascade_upperbody.xml')
        if not os.path.exists(xml):
            raise IOError(u"No such file: '%s'" % xml)
        cascade       = cv2.CascadeClassifier(xml)
        objects = list(cascade.detectMultiScale( smallImg,
            1.1, 3, 0
            #|cv.CV_HAAR_FIND_BIGGEST_OBJECT
            #|cv.CV_HAAR_DO_ROUGH_SEARCH
            |cv.CV_HAAR_SCALE_IMAGE,
            (30, 30) ))
        found += objects

        #t = time.time() - t
        #print("tdetection time = %gms\n", t*1000.)
        bbox = gtk.gdk.Rectangle(*(0,0,img.shape[1],img.shape[0]))
        # exclude duplicates (see also in 'classifyFeatures()')
        found_filtered = [gtk.gdk.Rectangle(*f) for f in self._util_merge_Regions(found, sub=True)[0]]
        result = []
        for i in range(len(found_filtered)):
            r = found_filtered[i]
            # the HOG detector returns slightly larger rectangles than the real objects.
            # so we slightly shrink the rectangles to get a nicer output.
            r.x += cv.Round(r.width*0.1)
            r.width = cv.Round(r.width*0.8)
            r.y += cv.Round(r.height*0.07)
            r.height = cv.Round(r.height*0.8)
            data = { 'ID':       (i+1), }
                     #'Center':   (int(r.x + r.width*0.5), int(r.y + r.height*0.5)), }
            # crop to image size (because of the slightly bigger boxes)
            r = bbox.intersect(r)
            #cv2.rectangle(img, (r.x, r.y), (r.x+r.width, r.y+r.height), cv.Scalar(0,255,0), 3)
            data['Position'] = tuple(np.int_(np.array(r)*scale))
            data['Coverage'] = float(data['Position'][2]*data['Position'][3])/(self.image_size[0]*self.image_size[1])
            result.append( data )
        #cv2.imshow("people detector", img)
        #c = cv2.waitKey(0) & 255

        self._info['People'] = result
        return

    def _detect_Geometry_CV(self):
        self._info['Geometry'] = []

        # skip file formats not supported (yet?)
        if (self.image_mime[1] in ['ogg', 'pdf', 'vnd.djvu']):
            return

        result = self._util_get_Geometry_CVnSCIPY()

        self._info['Geometry'] = [{'Lines': result['Lines'], 'Circles': result['Circles'], 'Corners': result['Corners'],
                                   'FFT_Comp': result['FFT_Comp'], 'SVD_Comp': result['SVD_Comp'], 'SVD_Min': result['SVD_Min']}]
        return

    # https://code.ros.org/trac/opencv/browser/trunk/opencv/samples/python/houghlines.py?rev=2770
    def _util_get_Geometry_CVnSCIPY(self):
        # http://docs.opencv.org/modules/imgproc/doc/feature_detection.html#cornerharris
        # http://docs.opencv.org/modules/imgproc/doc/feature_detection.html#houghcircles
        # http://docs.opencv.org/modules/imgproc/doc/feature_detection.html#houghlines
        # http://docs.opencv.org/modules/imgproc/doc/feature_detection.html#houghlinesp

        if hasattr(self, '_buffer_Geometry'):
            return self._buffer_Geometry

        self._buffer_Geometry = {'Lines': '-', 'Circles': '-', 'Edge_Ratio': '-', 'Corners': '-',
                                 'FFT_Comp': '-', 'FFT_Peaks': '-', 'SVD_Comp': '-', 'SVD_Min': '-'}

        scale = 1.
        try:
            img = cv2.imread(self.image_path_JPEG, 1)

            if (img == None):
                raise IOError

            # !!! the 'scale' here IS RELEVANT FOR THE DETECTION RATE;
            # how small and how many features are detected
            scale  = max([1., np.average(np.array(img.shape)[0:2]/500.)])
        except IOError:
            pywikibot.output(u'WARNING: unknown file type [_detect_Geometry_CV]')
            return self._buffer_Geometry
        except AttributeError:
            pywikibot.output(u'WARNING: unknown file type [_detect_Geometry_CV]')
            return self._buffer_Geometry

        # similar to face or people detection
        smallImg = np.empty( (cv.Round(img.shape[1]/scale), cv.Round(img.shape[0]/scale)), dtype=np.uint8 )
        _gray = cv2.cvtColor( img, cv.CV_BGR2GRAY )
        # smooth it, otherwise a lot of false circles may be detected
        #gray = cv2.GaussianBlur( _gray, (9, 9), 2 )
        gray = cv2.GaussianBlur( _gray, (5, 5), 2 )
        smallImg = cv2.resize( gray, smallImg.shape, interpolation=cv2.INTER_LINEAR )
        #smallImg = cv2.equalizeHist( smallImg )
        src = smallImg

        # https://code.ros.org/trac/opencv/browser/trunk/opencv/samples/python/houghlines.py?rev=2770
        #dst = cv2.Canny(src, 50, 200)
        dst = cv2.Canny(src, 10, 10)
        edges = cv2.Canny(src, 10, 10)
        #color_dst = cv2.cvtColor(dst, cv.CV_GRAY2BGR)

        # edges (in this sensitve form a meassure for color gradients)
        data = {}
        data['Edge_Ratio'] = float((edges != 0).sum())/(edges.shape[0]*edges.shape[1])

        # lines
        USE_STANDARD = True
        if USE_STANDARD:
            #lines = cv.HoughLines2(dst, storage, cv.CV_HOUGH_STANDARD, 1, pi / 180, 100, 0, 0)
            #lines = cv2.HoughLines(dst, 1, math.pi / 180, 100)
            lines = cv2.HoughLines(dst, 1, math.pi / 180, 200)
            if (lines is not None) and len(lines):
                lines = lines[0]
                data['Lines'] = len(lines)
            #for (rho, theta) in lines[:100]:
            #    a = math.cos(theta)
            #    b = math.sin(theta)
            #    x0 = a * rho 
            #    y0 = b * rho
            #    pt1 = (cv.Round(x0 + 1000*(-b)), cv.Round(y0 + 1000*(a)))
            #    pt2 = (cv.Round(x0 - 1000*(-b)), cv.Round(y0 - 1000*(a)))
            #    cv2.line(color_dst, pt1, pt2, cv.RGB(255, 0, 0), 3, 8)
        else:
            #lines = cv.HoughLines2(dst, storage, cv.CV_HOUGH_PROBABILISTIC, 1, pi / 180, 50, 50, 10)
            lines = cv2.HoughLinesP(dst, 1, math.pi / 180, 100) 
            #for line in lines:
            #    cv2.line(color_dst, line[0], line[1], cv.CV_RGB(255, 0, 0), 3, 8)

        # circles
        try:
            #circles = cv2.HoughCircles(src, cv.CV_HOUGH_GRADIENT, 2, src.shape[0]/4)#, 200, 100 )
            circles = cv2.HoughCircles(src, cv.CV_HOUGH_GRADIENT, 2, src.shape[0]/4, param2=200)
        except cv2.error:
            circles = None
        if (circles is not None) and len(circles):
            circles = circles[0]
            data['Circles'] = len(circles)
        #for c in circles:
        #    center = (cv.Round(c[0]), cv.Round(c[1]))
        #    radius = cv.Round(c[2])
        #    # draw the circle center
        #    cv2.circle( color_dst, center, 3, cv.CV_RGB(0,255,0), -1, 8, 0 )
        #    # draw the circle outline
        #    cv2.circle( color_dst, center, radius, cv.CV_RGB(0,0,255), 3, 8, 0 )

        # corners
        corner_dst = cv2.cornerHarris( edges, 2, 3, 0.04 )
        # Normalizing
        cv2.normalize( corner_dst, corner_dst, 0, 255, cv2.NORM_MINMAX, cv.CV_32FC1 )
        #dst_norm_scaled = cv2.convertScaleAbs( corner_dst )
        # Drawing a circle around corners
        corner = []
        for j in range(corner_dst.shape[0]):
            for i in range(corner_dst.shape[1]):
                if corner_dst[j,i] > 200:
                    #circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
                    corner.append( (j,i) )
        data['Corners'] = len(corner)

        #cv2.imshow("people detector", color_dst)
        #c = cv2.waitKey(0) & 255

        # fft
        gray = cv2.resize( _gray, smallImg.shape, interpolation=cv2.INTER_LINEAR )
        #s = (self.image_size[1], self.image_size[0])
        s = gray.shape
        fft = fftpack.fftn(gray)
        peaks = np.where(fft > (fft.max()*0.001))[0].shape[0]
        # shift quadrants so that low spatial frequencies are in the center
        #fft = fftpack.fftshift(fft)
        #fft = np.fft.fftn(gray)
        c = (np.array(s)/2.).astype(int)
        for i in range(0, min(c)-1, max( int(min(c)/50.), 1 )):
            fft[(c[0]-i):(c[0]+i+1),(c[1]-i):(c[1]+i+1)] = 0.
            #new = np.zeros(s)
            #new[(c[0]-i):(c[0]+i+1),(c[1]-i):(c[1]+i+1)] = fft[(c[0]-i):(c[0]+i+1),(c[1]-i):(c[1]+i+1)]
            #Image.fromarray(fftpack.fftshift(fft).real).show()
            ##Image.fromarray(fftpack.ifftn(fftpack.ifftshift(new)).real - gray).show()
            #Image.fromarray(fftpack.ifftn(fft).real - gray).show()
            if ((fftpack.ifftn(fft).real - gray).max() >= (255/2.)):
                break
        #fft = fftpack.ifftshift(fft)
        #Image.fromarray(fftpack.ifftn(fft).real).show()
        #Image.fromarray(np.fft.ifftn(fft).real).show()
        data['FFT_Comp']  = 1.-float(i*i)/(s[0]*s[1])
        data['FFT_Peaks'] = peaks
        #pywikibot.output( u'FFT_Comp: %s %s' % (1.-float(i*i)/(s[0]*s[1]), peaks) )

        # svd
        try:
            U, S, Vh = linalg.svd(np.matrix(gray))
            #U, S, Vh = linalg.svd(np.matrix(fft))      # do combined 'svd of fft'
            SS = np.zeros(s)
            ss = min(s)
            for i in range(0, len(S)-1, max( int(len(S)/100.), 1 )):   # (len(S)==ss) -> else; problem!
                #SS = np.zeros(s)
                #SS[:(ss-i),:(ss-i)] = np.diag(S[:(ss-i)])
                SS[:(i+1),:(i+1)] = np.diag(S[:(i+1)])
                #Image.fromarray(np.dot(np.dot(U, SS), Vh) - gray).show()
                #if ((np.dot(np.dot(U, SS), Vh) - gray).max() >= (255/4.)):
                if ((np.dot(np.dot(U, SS), Vh) - gray).max() < (255/4.)):
                    break
            #data['SVD_Comp'] = 1.-float(i)/ss
            data['SVD_Comp'] = float(i)/ss
            data['SVD_Min']  = S[:(i+1)].min()
            #pywikibot.output( u'SVD_Comp: %s' % (1.-float(i)/ss) )
            #pywikibot.output( u'SVD_Comp: %s %s %s' % (float(i)/ss, S[:(i+1)].min(), S[:(i+1)].max()) )
        except linalg.LinAlgError:
            # SVD did not converge; in fact this should NEVER happen...(?!?)
            pass

        if data:
            self._buffer_Geometry.update(data)
        return self._buffer_Geometry

    # .../opencv/samples/cpp/bagofwords_classification.cpp
    def _detectclassify_ObjectAll_CV(self):
        """Uses the 'The Bag of Words model' for detection and classification"""

        # http://app-solut.com/blog/2011/07/the-bag-of-words-model-in-opencv-2-2/
        # http://app-solut.com/blog/2011/07/using-the-normal-bayes-classifier-for-image-categorization-in-opencv/
        # http://authors.library.caltech.edu/7694/
        # http://www.vision.caltech.edu/Image_Datasets/Caltech256/
        # http://opencv.itseez.com/modules/features2d/doc/object_categorization.html
        
        # http://www.morethantechnical.com/2011/08/25/a-simple-object-classifier-with-bag-of-words-using-opencv-2-3-w-code/
        #   source: https://github.com/royshil/FoodcamClassifier
        # http://app-solut.com/blog/2011/07/using-the-normal-bayes-classifier-for-image-categorization-in-opencv/
        #   source: http://code.google.com/p/open-cv-bow-demo/downloads/detail?name=bowdemo.tar.gz&can=2&q=

        # parts of code here should/have to be placed into e.g. a own
        # class in 'dtbext/opencv/__init__.py' script/module
        
        self._info['Classify'] = []

        trained = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                   'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                   'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                   'sofa', 'train', 'tvmonitor',]
        bowDescPath = os.path.join(scriptdir, 'dtbext/opencv/data/bowImageDescriptors/000000.xml.gz')

        # https://code.ros.org/trac/opencv/browser/trunk/opencv/samples/cpp/bagofwords_classification.cpp?rev=3714
        # stand-alone (in shell) for training e.g. with:
        #   BoWclassify /data/toolserver/pywikipedia/dtbext/opencv/VOC2007 /data/toolserver/pywikipedia/dtbext/opencv/data FAST SURF BruteForce | tee run.log
        #   BoWclassify /data/toolserver/pywikipedia/dtbext/opencv/VOC2007 /data/toolserver/pywikipedia/dtbext/opencv/data HARRIS SIFT BruteForce | tee run.log
        # http://experienceopencv.blogspot.com/2011/02/object-recognition-bag-of-keypoints.html
        import dtbext.opencv as opencv

        if os.path.exists(bowDescPath):
            os.remove(bowDescPath)

        stdout = sys.stdout
        sys.stdout = StringIO.StringIO()
        #result = opencv.BoWclassify.main(0, '', '', '', '', '')
        result = opencv.BoWclassify.main(6, 
                                         os.path.join(scriptdir, 'dtbext/opencv/VOC2007'), 
                                         os.path.join(scriptdir, 'dtbext/opencv/data'), 
                                         'HARRIS',      # not important; given by training
                                         'SIFT',        # not important; given by training
                                         'BruteForce',  # not important; given by training
                                         [str(os.path.abspath(self.image_path).encode('latin-1'))])
        #out = sys.stdout.getvalue()
        sys.stdout = stdout
        #print out
        if not result:
            raise ImportError("BoW did not resolve; no results found!")
        os.remove(bowDescPath)

        # now make the algo working; confer also
        # http://www.xrce.xerox.com/layout/set/print/content/download/18763/134049/file/2004_010.pdf
        # http://people.csail.mit.edu/torralba/shortCourseRLOC/index.html

        self._info['Classify'] = [dict([ (trained[i], r) for i, r in enumerate(result) ])]
        return

    # a lot more paper and possible algos exist; (those with code are...)
    # http://www.lix.polytechnique.fr/~schwander/python-srm/
    # http://library.wolfram.com/infocenter/Demos/5725/#downloads
    # http://code.google.com/p/pymeanshift/wiki/Examples
    # (http://pythonvision.org/basic-tutorial, http://luispedro.org/software/mahotas, http://packages.python.org/pymorph/)
    def _detect_SegmentColors_JSEGnPIL(self):    # may be SLIC other other too...
        self._info['ColorRegions'] = []

        # skip file formats not supported (yet?)
        if (self.image_mime[1] in ['ogg', 'pdf', 'vnd.djvu']):
            return

        try:
            #im = Image.open(self.image_path).convert(mode = 'RGB')
            im = Image.open(self.image_path_JPEG)

            ## crop 25% of the image in order to give the bot a more human eye
            ## (needed for categorization only and thus should be done there/later)
            #scale  = 0.75    # crop 25% percent (area) bounding box
            #(w, h) = ( self.image_size[0]*math.sqrt(scale), self.image_size[1]*math.sqrt(scale) )
            #(l, t) = ( (self.image_size[0]-w)/2, (self.image_size[1]-h)/2 )
            #i = im.crop( (int(l), int(t), int(l+w), int(t+h)) )
            (l, t) = (0, 0)
            i = im
        except IOError:
            pywikibot.output(u'WARNING: unknown file type [_detect_SegmentColors_JSEGnPIL]')
            return

        result = []
        try:
            #h = i.histogram()   # average over WHOLE IMAGE
            (pic, scale) = self._util_detect_ColorSegments_JSEG(i)      # split image into segments first
            #(pic, scale) = self._util_detect_ColorSegments_SLIC(i)      # split image into superpixel first
            hist = self._util_get_ColorSegmentsHist_PIL(i, pic, scale)  #
            #pic  = self._util_merge_ColorSegments(pic, hist)            # iteratively in order to MERGE similar regions
            #(pic, scale_) = self._util_detect_ColorSegments_JSEG(pic)   # (final split)
            ##(pic, scale) = self._util_detect_ColorSegments_JSEG(pic)    # (final split)
            #hist = self._util_get_ColorSegmentsHist_PIL(i, pic, scale)  #
        except TypeError:
            pywikibot.output(u'WARNING: unknown file type [_detect_SegmentColors_JSEGnPIL]')
            return
        i = 0
        # (may be do an additional region merge according to same color names...)
        for (h, coverage, (center, bbox)) in hist:
            if (coverage < 0.05):    # at least 5% coverage needed (help for debugging/log_output)
                continue

            data = self._util_average_Color_colormath(h)
            data['Coverage'] = float(coverage)
            data['ID']       = (i+1)
            data['Center']   = (int(center[0]+l), int(center[1]+t))
            data['Position'] = (int(bbox[0]+l), int(bbox[1]+t), int(bbox[2]), int(bbox[3]))
            data['Delta_R']  = math.sqrt( (self.image_size[0]/2 - center[0])**2 + \
                                          (self.image_size[1]/2 - center[1])**2 )

            result.append( data )
            i += 1

        self._info['ColorRegions'] = result
        return

    # http://stackoverflow.com/questions/2270874/image-color-detection-using-python
    # https://gist.github.com/1246268
    # colormath-1.0.8/examples/delta_e.py, colormath-1.0.8/examples/conversions.py
    # http://code.google.com/p/python-colormath/
    # http://en.wikipedia.org/wiki/Color_difference
    # http://www.farb-tabelle.de/en/table-of-color.htm
    def _detect_AverageColor_PILnCV(self):
        self._info['ColorAverage'] = []

        # skip file formats not supported (yet?)
        if (self.image_mime[1] in ['ogg', 'pdf', 'vnd.djvu']):
            return

        try:
            # we need to have 3 channels (but e.g. grayscale 'P' has only 1)
            #i = Image.open(self.image_path).convert(mode = 'RGB')
            i = Image.open(self.image_path_JPEG)
            h = i.histogram()
        except IOError:
            pywikibot.output(u'WARNING: unknown file type [_detect_AverageColor_PILnCV]')
            return

        result              = self._util_average_Color_colormath(h)
        result['Gradient']  = self._util_get_Geometry_CVnSCIPY().get('Edge_Ratio', None) or '-'
        result['FFT_Peaks'] = self._util_get_Geometry_CVnSCIPY().get('FFT_Peaks', None) or '-'
        self._info['ColorAverage'] = [result]
        return

    def _detect_Properties_PIL(self):
        """Retrieve as much file property info possible, especially the same
           as commons does in order to compare if those libraries (ImageMagick,
           ...) are buggy (thus explicitely use other software for independence)"""
        #self.image_size = (None, None)
        self._info['Properties'] = [{'Format': u'-', 'Pages': 0}]
        if self.image_fileext == u'.svg':   # MIME: 'application/xml; charset=utf-8'
            # similar to PDF page count OR use BeautifulSoup
            svgcountpages = re.compile("<page>")
            pc = len(svgcountpages.findall( file(self.image_path,"r").read() ))

            #svg = rsvg.Handle(self.image_path)

            # http://validator.w3.org/docs/api.html#libs
            # http://pypi.python.org/pypi/py_w3c/
            vld = HTMLValidator()
            valid = u'SVG'
            try:
                vld.validate(self.image.fileUrl())
                valid = (u'Valid SVG' if vld.result.validity == 'true' else u'Invalid SVG')
            except urllib2.URLError:
                pass
            except ValidationFault:
                pass
            #print vld.errors, vld.warnings

            #self.image_size = (svg.props.width, svg.props.height)

            result = { 'Format':     valid,
                       'Mode':       u'-',
                       'Palette':    u'-',
                       'Pages':      pc, }
            # may be set {{validSVG}} also or do something in bot template to
            # recognize 'Format=SVG (valid)' ...
        elif self.image_mime[1] == 'pdf':   # MIME: 'application/pdf; charset=binary'
            # http://code.activestate.com/recipes/496837-count-pdf-pages/
            #rxcountpages = re.compile(r"$\s*/Type\s*/Page[/\s]", re.MULTILINE|re.DOTALL)
            rxcountpages = re.compile(r"/Type\s*/Page([^s]|$)", re.MULTILINE|re.DOTALL)    # PDF v. 1.3,1.4,1.5,1.6
            pc = len(rxcountpages.findall( file(self.image_path,"rb").read() ))

            result = { 'Format':     u'PDF',
                       'Mode':       u'-',
                       'Palette':    u'-',
                       'Pages':      pc, }
        elif (self.image_mime[0] == 'image') and \
             (self.image_mime[1] != 'vnd.djvu'):   # MIME: 'image/jpeg; charset=binary', ...
            try:
                i = Image.open(self.image_path)
            except IOError:
                pywikibot.output(u'WARNING: unknown (image) file type [_detect_Properties_PIL]')
                return

            # http://mail.python.org/pipermail/image-sig/1999-May/000740.html
            pc=0         # count number of pages
            while True:
                try:
                    i.seek(pc)
                except EOFError:
                    break
                pc+=1
            i.seek(0)    # restore default

            # http://grokbase.com/t/python/image-sig/082psaxt6k/embedded-icc-profiles
            # python-lcms (littlecms) may be freeimage library
            #icc = i.app['APP2']     # jpeg
            #icc = i.tag[34675]      # tiff
            #icc = re.sub('[^%s]'%string.printable, ' ', icc)
            ## more image formats and more post-processing needed...

            #self.image_size = i.size

            result = { #'bands':      i.getbands(),
                       #'bbox':       i.getbbox(),
                       'Format':     i.format,
                       'Mode':       i.mode,
                       #'info':       i.info,
                       #'stat':       os.stat(self.image_path),
                       'Palette':    str(len(i.palette.palette)) if i.palette else u'-',
                       'Pages':      pc, }
        elif self.image_mime[1] == 'ogg':   # MIME: 'application/ogg; charset=binary'
            # 'ffprobe' (ffmpeg); audio and video streams files (ogv, oga, ...)
            d = self._util_get_DataStreams_FFMPEG()
            #print d
            result = { 'Format': u'%s' % d['format']['format_name'].upper() }
        #elif self.image_mime[0] =='audio':  # MIME: 'audio/midi; charset=binary'
        #    result = {}
        # djvu: python-djvulibre or python-djvu for djvu support
        # http://pypi.python.org/pypi/python-djvulibre/0.3.9
        #elif self.image_fileext == u'.xcf'
        #    result = {}
        #    # DO NOT use ImageMagick (identify) instead of PIL to get these info !!
        else:
            pywikibot.output(u'WARNING: unknown (generic) file type [_detect_Properties_PIL]')
            return

        result['Dimensions'] = self.image_size
        result['Filesize']   = os.path.getsize(self.image_path)
        result['MIME']       = u'%s/%s' % tuple(self.image_mime[:2])

        #self._info['Properties'] = [result]
        self._info['Properties'][0].update(result)
        return

    # http://stackoverflow.com/questions/2270874/image-color-detection-using-python
    # https://gist.github.com/1246268
    # colormath-1.0.8/examples/delta_e.py, colormath-1.0.8/examples/conversions.py
    # http://code.google.com/p/python-colormath/
    # http://en.wikipedia.org/wiki/Color_difference
    # http://www.farb-tabelle.de/en/table-of-color.htm
    # http://www5.konicaminolta.eu/de/messinstrumente/color-light-language.html
    def _util_average_Color_colormath(self, h):
        # split into red, green, blue
        r = h[0:256]
        g = h[256:256*2]
        b = h[256*2: 256*3]
        
        # perform the weighted average of each channel:
        # the *index* 'i' is the channel value, and the *value* 'w' is its weight
        rgb = (
                sum( i*w for i, w in enumerate(r) ) / max(1, sum(r)),
                sum( i*w for i, w in enumerate(g) ) / max(1, sum(g)),
                sum( i*w for i, w in enumerate(b) ) / max(1, sum(b))
        )

        # count number of colors used more than 1% of maximum
        ma    = 0.01*max(h)
        count = sum([int(c > ma) for c in h])

        data = { #'histogram': h,
                 'RGB':   rgb,
                 'Peaks': float(count)/len(h), }

        #colors = pycolorname.RAL.colors
        #colors = pycolorname.pantone.Formula_Guide_Solid
        colors = pycolorname.pantone.Fashion_Home_paper
        
        #print "=== RGB Example: RGB->LAB ==="
        # Instantiate an Lab color object with the given values.
        rgb = RGBColor(rgb[0], rgb[1], rgb[2], rgb_type='sRGB')
        # Show a string representation.
        #print rgb
        # Convert RGB to LAB using a D50 illuminant.
        lab = rgb.convert_to('lab', target_illuminant='D65')
        #print lab
        #print "=== End Example ===\n"
        
        # Reference color.
        #color1 = LabColor(lab_l=0.9, lab_a=16.3, lab_b=-2.22)
        # Color to be compared to the reference.
        #color2 = LabColor(lab_l=0.7, lab_a=14.2, lab_b=-1.80)
        color2 = lab

        res = (1.E100, '')
        for c in colors:
            rgb = colors[c]
            rgb = RGBColor(rgb[0], rgb[1], rgb[2], rgb_type='sRGB')
            color1 = rgb.convert_to('lab', target_illuminant='D65')

            #print "== Delta E Colors =="
            #print " COLOR1: %s" % color1
            #print " COLOR2: %s" % color2
            #print "== Results =="
            #print " CIE2000: %.3f" % color1.delta_e(color2, mode='cie2000')
            ## Typically used for acceptability.
            #print "     CMC: %.3f (2:1)" % color1.delta_e(color2, mode='cmc', pl=2, pc=1)
            ## Typically used to more closely model human percetion.
            #print "     CMC: %.3f (1:1)" % color1.delta_e(color2, mode='cmc', pl=1, pc=1)

            r = color1.delta_e(color2, mode='cmc', pl=2, pc=1)
            if (r < res[0]):
                res = (r, c, colors[c])
        data['Color']   = res[1]
        data['Delta_E'] = res[0]
        data['RGBref']  = res[2]

        return data

    def _util_detect_ColorSegments_JSEG(self, im):
        tmpjpg = os.path.join(scriptdir, "cache/jseg_buf.jpg")
        tmpgif = os.path.join(scriptdir, "cache/jseg_buf.gif")

        # same scale func as in '_detect_Faces_CV'
        scale  = max([1., np.average(np.array(im.size)[0:2]/200.)])
        #print np.array(im.size)/scale, scale
        try:
            smallImg = im.resize( tuple(np.int_(np.array(im.size)/scale)), Image.ANTIALIAS )
        except IOError:
            pywikibot.output(u'WARNING: unknown file type [_util_detect_ColorSegments_JSEG]')
            return
        
        #im.thumbnail(size, Image.ANTIALIAS) # size is 640x480
        smallImg.convert('RGB').save(tmpjpg, "JPEG", quality=100, optimize=True)
        
        # Program limitation: The total number of regions in the image must be less
        # than 256 before the region merging process. This works for most images
        # smaller than 512x512.
        
        # Processing time will be about 10 seconds for an 192x128 image and 60 seconds
        # for a 352x240 image. It will take several minutes for a 512x512 image.
        # Minimum image size is 64x64.
        
        # ^^^  THUS RESCALING TO ABOUT 200px ABOVE  ^^^

        # sys.stdout handeled, but with freopen which could give issues
        import dtbext.jseg as jseg
        # e.g. "segdist -i test3.jpg -t 6 -r9 test3.map.gif"
        enable_recovery()   # enable recovery from hard crash
        jseg.segdist_cpp.main( ("segdist -i %s -t 6 -r9 %s"%(tmpjpg, tmpgif)).split(" ") )
        disable_recovery()  # disable since everything worked out fine
        #out = open((tmpgif + ".stdout"), "r").read()    # reading stdout
        #print out
        os.remove(tmpgif + ".stdout")
        
        os.remove( tmpjpg )
        
        # http://stackoverflow.com/questions/384759/pil-and-numpy
        pic = Image.open(tmpgif)
        #pix = np.array(pic)
        #Image.fromarray(10*pix).show()
        
        os.remove( tmpgif )

        return (pic, scale)

    # http://planet.scipy.org/
    # http://peekaboo-vision.blogspot.ch/2012/05/superpixels-for-python-pretty-slic.html
    # http://ivrg.epfl.ch/supplementary_material/RK_SLICSuperpixels/index.html
    def _util_detect_ColorSegments_SLIC(self, img):
        import dtbext.slic as slic

        im = np.array(img)
        image_argb = np.dstack([im[:, :, :1], im]).copy("C")
        #region_labels = slic.slic_n(image_argb, 1000, 10)
        region_labels = slic.slic_n(image_argb, 1000, 50)
        slic.contours(image_argb, region_labels, 10)
        #import matplotlib.pyplot as plt
        #plt.imshow(image_argb[:, :, 1:].copy())
        #plt.show()

        #pic = Image.fromarray(region_labels)
        #pic.show()

        #return (pic, 1.)
        return (region_labels, 1.)

    def _util_get_ColorSegmentsHist_PIL(self, im, pic, scale):
        if not (type(np.ndarray(None)) == type(pic)):
            pix = np.array(pic)
            #Image.fromarray(10*pix).show()
        else:
            pix = pic
            #Image.fromarray(255*pix/np.max(pix)).show()

        try:
            smallImg = im.resize( tuple(np.int_(np.array(im.size)/scale)), Image.ANTIALIAS )
        except IOError:
            pywikibot.output(u'WARNING: unknown file type [_util_get_ColorSegmentsHist_PIL]')
            return

        imgsize = float(smallImg.size[0]*smallImg.size[1])
        hist = []
        for i in range(np.max(pix)+1):
            mask   = np.uint8(pix == i)*255
            (y, x) = np.where(mask != 0)
            center = (np.average(x)*scale, np.average(y)*scale)
            bbox   = (np.min(x)*scale, np.min(y)*scale, 
                      (np.max(x)-np.min(x))*scale, (np.max(y)-np.min(y))*scale)
            #coverage = np.count_nonzero(mask)/imgsize
            coverage = (mask != 0).sum()/imgsize    # count_nonzero is missing in older numpy
            mask = Image.fromarray( mask )
            h    = smallImg.histogram(mask)
            #smallImg.show()
            #dispImg = Image.new('RGBA', smallImg.size)
            #dispImg.paste(smallImg, mask)
            #dispImg.show()
            if (len(h) == 256):
                pywikibot.output(u"gray scale image, try to fix...")
                h = h*3
            if (len(h) == 256*4):
                pywikibot.output(u"4-ch. image, try to fix (exclude transparency)...")
                h = h[0:(256*3)]
            hist.append( (h, coverage, (center, bbox)) )
        
        return hist

    # http://www.scipy.org/SciPyPackages/Ndimage
    # http://www.pythonware.com/library/pil/handbook/imagefilter.htm
    def _util_merge_ColorSegments(self, im, hist):
        # merge regions by simplifying through average color and re-running
        # JSEG again...

        if not (type(np.ndarray(None)) == type(im)):
            pix = np.array(im)
        else:
            pix = im
            im  = Image.fromarray(255*pix/np.max(pix))

        im = im.convert('RGB')

        for j, (h, coverage, (center, bbox)) in enumerate(hist):
            # split into red, green, blue
            r = h[0:256]
            g = h[256:256*2]
            b = h[256*2: 256*3]
            
            # perform the weighted average of each channel:
            # the *index* 'i' is the channel value, and the *value* 'w' is its weight
            rgb = (
                    sum( i*w for i, w in enumerate(r) ) / max(1, sum(r)),
                    sum( i*w for i, w in enumerate(g) ) / max(1, sum(g)),
                    sum( i*w for i, w in enumerate(b) ) / max(1, sum(b))
            )
            # color frequency analysis; do not average regions with high fluctations
            #rgb2 = (
            #        sum( i*i*w for i, w in enumerate(r) ) / max(1, sum(r)),
            #        sum( i*i*w for i, w in enumerate(g) ) / max(1, sum(g)),
            #        sum( i*i*w for i, w in enumerate(b) ) / max(1, sum(b))
            #)
            #if ( 500. < np.average( (
            #       rgb2[0] - rgb[0]**2,
            #       rgb2[1] - rgb[1]**2,
            #       rgb2[2] - rgb[2]**2, ) ) ):
            #           continue

            mask = np.uint8(pix == j)*255
            mask = Image.fromarray( mask )
            #dispImg = Image.new('RGB', im.size)
            #dispImg.paste(rgb, mask=mask)
            #dispImg.show()
            im.paste(rgb, mask=mask)

        pix = np.array(im)
        pix[:,:,0] = ndimage.gaussian_filter(pix[:,:,0], .5)
        pix[:,:,1] = ndimage.gaussian_filter(pix[:,:,1], .5)
        pix[:,:,2] = ndimage.gaussian_filter(pix[:,:,2], .5)
        im = Image.fromarray( pix, mode='RGB' )
        #im = im.filter(ImageFilter.BLUR)   # or 'SMOOTH'

        return im

    # Category:...      (several; look at self.gatherFeatures for more hints)
    def _detect_Trained_CV(self, info_desc, cascade_file, maxdim=500.):
        # general (self trained) classification (e.g. people, ...)
        # http://www.computer-vision-software.com/blog/2009/11/faq-opencv-haartraining/

        # Can be used with haar classifier (use: opencv_haartraining) and
        # cascaded classifier (use: opencv_traincascade), both should work.

        # !!! train a own cascade classifier like for face detection used
        # !!! with 'opencv_haartraing' -> xml (file to use like in face/eye detection)

        # analogue to face detection:

        self._info[info_desc] = []

        # skip file formats not supported (yet?)
        if (self.image_mime[1] in ['ogg', 'pdf', 'vnd.djvu']):
            return

        # http://tutorial-haartraining.googlecode.com/svn/trunk/data/haarcascades/
        # or own xml files trained onto specific file database/set
        xml = os.path.join(scriptdir, ('dtbext/opencv/haarcascades/' + cascade_file))
        if not os.path.exists(xml):
            raise IOError(u"No such file: '%s'" % xml)
        cascade       = cv2.CascadeClassifier(xml)

        scale = 1.
        try:
            img    = cv2.imread( self.image_path_JPEG, 1 )
            if (img == None) or (self.image_size[0] is None):
                raise IOError
            
            # !!! the 'scale' here IS RELEVANT FOR THE DETECTION RATE;
            # how small and how many features are detected
            scale  = max([1., np.average(np.array(img.shape)[0:2]/maxdim)])
        except IOError:
            pywikibot.output(u'WARNING: unknown file type [_detect_Trained_CV]')
            return
        except AttributeError:
            pywikibot.output(u'WARNING: unknown file type [_detect_Trained_CV]')
            return

        # similar to face detection
        smallImg = np.empty( (cv.Round(img.shape[1]/scale), cv.Round(img.shape[0]/scale)), dtype=np.uint8 )
        gray = cv2.cvtColor( img, cv.CV_BGR2GRAY )
        smallImg = cv2.resize( gray, smallImg.shape, interpolation=cv2.INTER_LINEAR )
        smallImg = cv2.equalizeHist( smallImg )

        objects = list(cascade.detectMultiScale( smallImg,
            1.1, 5, 0
            #|cv.CV_HAAR_FIND_BIGGEST_OBJECT
            #|cv.CV_HAAR_DO_ROUGH_SEARCH
            |cv.CV_HAAR_SCALE_IMAGE,
            (30, 30) ))

        result = []
        for i, r in enumerate(objects):
            data = { 'Position': tuple(np.int_(np.array(r)*scale)) }
            data['Coverage'] = float(data['Position'][2]*data['Position'][3])/(self.image_size[0]*self.image_size[1])
            result.append( data )

        # generic detection ...

        self._info[info_desc] = result
        return

    # ./run-test (ocropus/ocropy)
    # (in fact all scripts/executables used here are pure python scripts!!!)
    def _recognize_OpticalText_ocropus(self):
        # optical text recognition (tesseract & ocropus, ...)
        # (no full recognition but - at least - just classify as 'contains text')
        # http://www.claraocr.org/de/ocr/ocr-software/open-source-ocr.html
        # https://github.com/edsu/ocropy
        # http://de.wikipedia.org/wiki/Benutzer:DrTrigonBot/Doku#Categorization
        # Usage:tesseract imagename outputbase [-l lang] [configfile [[+|-]varfile]...]
        # tesseract imagename.tif output

        # (it's simpler to run the scripts/executables in own environment/interpreter...)

        # skip file formats not supported (yet?)
        if (self.image_mime[1] in ['ogg', 'pdf', 'vnd.djvu']):
            return

        path = os.path.join(scriptdir, 'dtbext/_ocropus/ocropy')

        curdir = os.path.abspath(os.curdir)
        os.chdir(path)

        # binarization
        if os.path.exists(os.path.join(path, "temp")):
            shutil.rmtree(os.path.join(path, "temp"))
        if os.system("ocropus-nlbin %s -o %s" % (self.image_path_JPEG, os.path.join(path, "temp"))):
            raise ImportError("ocropus not found!")
        
        # page level segmentation
        if os.system("ocropus-gpageseg --minscale 6.0 '%s'" % os.path.join(path, "temp/????.bin.png")):
            # detection error
            return
        
        # raw text line recognition
        if os.system("ocropus-lattices --writebestpath '%s'" % os.path.join(path, "temp/????/??????.bin.png")):
            # detection error
            return
        
        # language model application
        # (optional - improve the raw results by applying a pretrained model)
        os.environ['OCROPUS_DATA'] = os.path.join(path, "models/")
        if os.system("ocropus-ngraphs '%s'" % os.path.join(path, "temp/????/??????.lattice")):
            # detection error
            return
        
        # create hOCR output
        if os.system("ocropus-hocr '%s' -o %s" % (os.path.join(path, "temp/????.bin.png"), os.path.join(path, "temp.html"))):
            # detection error
            return
        
        ## 'create HTML for debugging (use "firefox temp/index.html" to view)'
        ## (optional - generate human readable debug output)
        #if os.system("ocropus-visualize-results %s" % os.path.join(path, "temp")):
        #    # detection error
        #    return
        
        # "to see recognition results, type: firefox temp.html"
        # "to see details on the recognition process, type: firefox temp/index.html"
        tmpfile = open(os.path.join(path, "temp.html"), 'r')
        data = tmpfile.read()
        tmpfile.close()

        shutil.rmtree(os.path.join(path, "temp"))
        os.remove(os.path.join(path, "temp.html"))

        os.chdir(curdir)

        #print data
        pywikibot.output(data)
 
    def _detect_EmbeddedText_poppler(self):
        # may be also: http://www.reportlab.com/software/opensource/rl-toolkit/

        self._info['Text'] = []

        if not (self.image_mime[1] == 'pdf'):
            return

        # poppler pdftotext/pdfimages
        # (similar as in '_util_get_DataTags_EXIF' but with stderr and no json output)
        # http://poppler.freedesktop.org/
        # http://www.izzycode.com/bash/how-to-install-pdf2text-on-centos-fedora-redhat.html
        # MIGHT BE BETTER TO USE AS PYTHON MODULE:
        # https://launchpad.net/poppler-python/
        # http://stackoverflow.com/questions/2732178/extracting-text-from-pdf-with-poppler-c
        # http://stackoverflow.com/questions/25665/python-module-for-converting-pdf-to-text
        #proc = Popen("pdftotext -layout %s %s" % (self.image_path, self.image_path+'.txt'), 
        proc = Popen("pdftotext %s %s" % (self.image_path, self.image_path+'.txt'), 
                     shell=True, stderr=PIPE)#.stderr.readlines()
        proc.wait()
        if proc.returncode:
            raise ImportError("pdftotext not found!")
        data = open(self.image_path+'.txt', 'r').readlines()
        os.remove( self.image_path+'.txt' )

#        self._content_text = data
        (s1, l1) = (len(u''.join(data)), len(data))

        tmp_path = os.path.join(os.environ.get('TMP', '/tmp'), 'DrTrigonBot/')
        os.mkdir( tmp_path )
# switch this part off since 'pdfimages' (on toolserver) is too old; TS-1449
#        proc = Popen("pdfimages -p %s %s/" % (self.image_path, tmp_path), 
        proc = Popen("pdfimages %s %s/" % (self.image_path, tmp_path), 
                     shell=True, stderr=PIPE)#.stderr.readlines()
        proc.wait()
        if proc.returncode:
            raise ImportError("pdfimages not found!")
        images = os.listdir( tmp_path )
#        pages  = set()
        for f in images:
#            pages.add( int(f.split('-')[1]) )
            os.remove( os.path.join(tmp_path, f) )
        os.rmdir( tmp_path )
        
        ## pdfminer (tools/pdf2txt.py)
        ## http://denis.papathanasiou.org/?p=343 (for layout and images)
        #debug = 0
        #laparams = layout.LAParams()
        ##
        #pdfparser.PDFDocument.debug        = debug
        #pdfparser.PDFParser.debug          = debug
        #cmapdb.CMapDB.debug                = debug
        #pdfinterp.PDFResourceManager.debug = debug
        #pdfinterp.PDFPageInterpreter.debug = debug
        #pdfdevice.PDFDevice.debug          = debug
        ##
        #rsrcmgr = pdfinterp.PDFResourceManager(caching=True)
        #outfp = StringIO.StringIO()
        #device = converter.TextConverter(rsrcmgr, outfp, codec='utf-8', laparams=laparams)
        ##device = converter.XMLConverter(rsrcmgr, outfp, codec='utf-8', laparams=laparams, outdir=None)
        ##device = converter.HTMLConverter(rsrcmgr, outfp, codec='utf-8', scale=1,
        ##                       layoutmode='normal', laparams=laparams, outdir=None)
        ##device = pdfdevice.TagExtractor(rsrcmgr, outfp, codec='utf-8')
        #fp = file(self.image_path, 'rb')
        #try:
        #    pdfinterp.process_pdf(rsrcmgr, device, fp, set(), maxpages=0, password='',
        #                caching=True, check_extractable=False)
        #except AssertionError:
        #    pywikibot.output(u'WARNING: pdfminer missed, may be corrupt [_detect_EmbeddedText_poppler]')
        #    return
        #except TypeError:
        #    pywikibot.output(u'WARNING: pdfminer missed, may be corrupt [_detect_EmbeddedText_poppler]')
        #    return
        #fp.close()
        #device.close()
        #data = outfp.getvalue().splitlines(True)
        #
        #(s2, l2) = (len(u''.join(data)), len(data))

        result = { 'Size':     s1,
                   'Lines':    l1,
                   #'Data':     data,
                   #'Position': pos,
#                   'Images':   u'%s (on %s page(s))' % (len(images), len(list(pages))),  # pages containing images
                   'Images':   u'%s' % len(images),
                   'Type':     u'-', }  # 'Type' could be u'OCR' above...

        self._info['Text'] = [result]

        return

    def _recognize_OpticalCodes_dmtxNzbar(self):
        # barcode and Data Matrix recognition (libdmtx/pydmtx, zbar, gocr?)
        # http://libdmtx.wikidot.com/libdmtx-python-wrapper
        # http://blog.globalstomp.com/2011/09/decoding-qr-code-code-128-code-39.html
        # http://zbar.sourceforge.net/
        # http://pypi.python.org/pypi/zbar
        
        self._info['OpticalCodes'] = []

        # skip file formats not supported (yet?)
        if (self.image_mime[1] in ['ogg', 'pdf', 'vnd.djvu']):
            return

        # DataMatrix
        try:
            from pydmtx import DataMatrix           # linux distro package (fedora)
        except:
            from dtbext._pydmtx import DataMatrix   # TS (debian)

        ## Write a Data Matrix barcode
        #dm_write = DataMatrix()
        #dm_write.encode("Hello, world!")
        #dm_write.save("hello.png", "png")

        scale = 1.
        try:
            # Read a Data Matrix barcode
            dm_read = DataMatrix()
            img = Image.open(self.image_path_JPEG)
            
            # http://libdmtx.wikidot.com/libdmtx-python-wrapper
            if img.mode != 'RGB':
               img = img.convert('RGB')

            scale  = max([1., np.average(np.array(img.size)/200.)])
        except IOError:
            pywikibot.output(u'WARNING: unknown file type [_recognize_OpticalCodes_dmtxNzbar]')
            return
        
        smallImg = img.resize( (int(img.size[0]/scale), int(img.size[1]/scale)) )
        img = smallImg

        enable_recovery()   # enable recovery from hard crash
        #res = dm_read.decode(img.size[0], img.size[1], buffer(img.tostring()))
        disable_recovery()  # disable since everything worked out fine
        #print res

        result = []
        i      = -1
        for i in range(dm_read.count()):
            data, bbox = dm_read.stats(i+1)
            bbox = np.array(bbox)
            x, y = bbox[:,0], bbox[:,1]
            pos  = (np.min(x), np.min(y), np.max(x)-np.min(x), np.max(y)-np.min(y))
            result.append({ 'ID':       (i+1),
                            #'Data':     dm_read.message(i+1),
                            'Data':     data,
                            'Position': pos,
                            'Type':     u'DataMatrix',
                            'Quality':  10, })
        
        self._info['OpticalCodes'] = result

        # supports many popular symbologies
        try:
            import zbar                     # TS (debian)
        except:
            import dtbext._zbar as zbar     # other distros (fedora)
        
        try:
            img = Image.open(self.image_path_JPEG).convert('L')
            width, height = img.size
        except IOError:
            pywikibot.output(u'WARNING: unknown file type [_recognize_OpticalCodes_dmtxNzbar]')
            return
        
        scanner = zbar.ImageScanner()
        scanner.parse_config('enable')
        zbar_img = zbar.Image(width, height, 'Y800', img.tostring())
        
        # scan the image for barcodes
        # http://zbar.sourceforge.net/api/zbar_8h.html
        scanner.scan(zbar_img)

        for symbol in zbar_img:
            i += 1
            p = np.array(symbol.location)   # list of points within code region/area
            p = (min(p[:,0]), min(p[:,1]), (max(p[:,0])-min(p[:,0])), (max(p[:,1])-min(p[:,1])))
            result.append({ #'components': symbol.components,
                            'ID':         (i+1),
                            #'Count':      symbol.count,         # 'ID'?
                            'Data':       symbol.data or u'-',
                            'Position':   p,                    # (left, top, width, height)
                            'Quality':    symbol.quality,       # usable for 'Confidence'
                            'Type':       symbol.type, })
        
        # further detection ?

        self._info['OpticalCodes'] = result
        return

    def _detect_Chessboard_CV(self):
        # Chessboard (opencv reference detector)
        # http://nullege.com/codes/show/src%40o%40p%40opencvpython-HEAD%40samples%40chessboard.py/12/cv.FindChessboardCorners/python

        self._info['Chessboard'] = []

        # skip file formats not supported (yet?)
        if (self.image_mime[1] in ['ogg', 'pdf', 'vnd.djvu']):
            return

        scale = 1.
        try:
            #cv.NamedWindow("win")
            #im = cv.LoadImage(self.image_path_JPEG, cv.CV_LOAD_IMAGE_GRAYSCALE)
            ##im3 = cv.LoadImage(self.image_path_JPEG, cv.CV_LOAD_IMAGE_COLOR)
            im = cv2.imread( self.image_path_JPEG, cv2.CV_LOAD_IMAGE_GRAYSCALE )
            chessboard_dim = ( 7, 7 )
            if im == None:
                raise IOError

            scale  = max([1., np.average(np.array(im.shape)[0:2]/1000.)])
        except IOError:
            pywikibot.output(u'WARNING: unknown file type [_detect_Chessboard_CV]')
            return
        except AttributeError:
            pywikibot.output(u'WARNING: unknown file type [_detect_Chessboard_CV]')
            return

        smallImg = np.empty( (cv.Round(im.shape[1]/scale), cv.Round(im.shape[0]/scale)), dtype=np.uint8 )
        #gray = cv2.cvtColor( im, cv.CV_BGR2GRAY )
        smallImg = cv2.resize( im, smallImg.shape, interpolation=cv2.INTER_LINEAR )
        #smallImg = cv2.equalizeHist( smallImg )
        im = smallImg

        found_all = False
        corners   = None
        try:
            #found_all, corners = cv.FindChessboardCorners( im, chessboard_dim )
            found_all, corners = cv2.findChessboardCorners( im, chessboard_dim )
        except cv2.error, e:
            pywikibot.output(u'%s' % e)

        #cv2.drawChessboardCorners( im, chessboard_dim, corners, found_all )
        ##cv2.imshow("win", im)
        ##cv2.waitKey()

        if corners is not None:
            corners = [ tuple(item[0]) for item in corners ]
            self._info['Chessboard'] = [{ 'Corners': corners, }]

#        # chess board recognition (more tolerant)
#        # http://codebazaar.blogspot.ch/2011/08/chess-board-recognition-project-part-1.html
#        # https://code.ros.org/trac/opencv/browser/trunk/opencv/samples/python/houghlines.py?rev=2770
#        # http://docs.opencv.org/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html
#        dst = im.copy()
#        color_dst = cv2.cvtColor(dst, cv.CV_GRAY2BGR)
#        dst = cv2.GaussianBlur(dst, (3, 3), 5)
#        thr = 150
#        dst = cv2.Canny(dst, thr, 3*thr)
#        cv2.imshow("win", dst)
#        cv2.waitKey()
#        # lines to find grid
#        # http://dsp.stackexchange.com/questions/2420/alternatives-to-hough-transform-for-detecting-a-grid-like-structure
#        USE_STANDARD = True
#        if USE_STANDARD:
#            #lines = cv.HoughLines2(dst, storage, cv.CV_HOUGH_STANDARD, 1, pi / 180, 100, 0, 0)
#            #lines = cv2.HoughLines(dst, 1, math.pi / 180, 100)
#            lines = cv2.HoughLines(dst, 1, math.pi / 180, 150)
#            if (lines is not None) and len(lines):
#                lines = lines[0]
#                #data['Lines'] = len(lines)
#
#            ls = np.array(lines)
#            import pylab
#            (n, bins, patches) = pylab.hist(ls[:,1])
#            print n, bins, patches
#            pylab.grid(True)
#            pylab.show()
#
#            for (rho, theta) in lines:
#                #if theta > 0.3125: continue
#                a = math.cos(theta)
#                b = math.sin(theta)
#                x0 = a * rho 
#                y0 = b * rho
#                pt1 = (cv.Round(x0 + 1000*(-b)), cv.Round(y0 + 1000*(a)))
#                pt2 = (cv.Round(x0 - 1000*(-b)), cv.Round(y0 - 1000*(a)))
#                cv2.line(color_dst, pt1, pt2, cv.RGB(255, 0, 0), 3, 8)
#        else:
#            #lines = cv.HoughLines2(dst, storage, cv.CV_HOUGH_PROBABILISTIC, 1, pi / 180, 50, 50, 10)
#            lines = cv2.HoughLinesP(dst, 1, math.pi / 180, 100) 
#
#            for line in lines[0]:
#                print line
#                cv2.line(color_dst, tuple(line[0:2]), tuple(line[2:4]), cv.CV_RGB(255, 0, 0), 3, 8)
#        cv2.imshow("win", color_dst)
#        cv2.waitKey()

        if found_all:
            # pose detection
            # http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
            # http://stackoverflow.com/questions/10022568/opencv-2-3-camera-calibration
            d = shelve.open( os.path.join(scriptdir, 'dtbext/opencv/camera_virtual_default') )
            if ('retval' not in d):
                # http://commons.wikimedia.org/wiki/File:Mutilated_checkerboard_3.jpg
                pywikibot.output(u"Doing (virtual) camera calibration onto reference image 'File:Mutilated_checkerboard_3.jpg'")
                im3 = cv2.imread( 'Mutilated_checkerboard_3.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE )
                im3 = cv2.resize( im3, (cv.Round(im3.shape[1]/scale), cv.Round(im3.shape[0]/scale)), interpolation=cv2.INTER_LINEAR )
                # Compute the the three dimensional world-coordinates
                tmp = []
                for h in range(chessboard_dim[0]):
                    for w in range(chessboard_dim[1]):
                        tmp.append( (float(h), float(w), 0.0) )
                objectPoints = np.array(tmp)
                # Compute matrices
                _found_all, _corners = cv2.findChessboardCorners( im3, chessboard_dim, flags=cv.CV_CALIB_CB_ADAPTIVE_THRESH | cv.CV_CALIB_CB_FILTER_QUADS )
                #cv2.drawChessboardCorners( im3, chessboard_dim, _corners, _found_all )
                retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera([objectPoints.astype('float32')], [_corners.astype('float32')], im3.shape, np.eye(3), np.zeros((5, 1)))
                fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(cameraMatrix, im3.shape, 1.0, 1.0)
                d['objectPoints']   = [objectPoints.astype('float32')]  # shape: (49, 3)    in a list of 1 item
                d['imagePoints']    = [_corners.astype('float32')]      # shape: (49, 1, 2) in a list of 1 item
                d['cameraMatrix']   = cameraMatrix
                d['distCoeffs']     = distCoeffs
                d['rvecs']          = rvecs
                d['tvecs']          = tvecs
                d['imageSize']      = im3.shape
                d['apertureWidth']  = 1.0
                d['apertureHeight'] = 1.0
                d['fovx']           = fovx
                d['fovy']           = fovy
                d['focalLength']    = focalLength
                d['principalPoint'] = principalPoint
                d['aspectRatio']    = aspectRatio
                d['retval']         = retval
            else:
                objectPoints = d['objectPoints'][0]
                cameraMatrix, distCoeffs = d['cameraMatrix'], d['distCoeffs']
                # would be nice to use these:
                #cameraMatrix, distCoeffs = np.eye(3), np.zeros((5,1))
                # ..,since they are simple... else other have to be documented as "used calibration" !!!
            d.close()
            # http://answers.opencv.org/question/1073/what-format-does-cv2solvepnp-use-for-points-in/
            rvec, tvec = cv2.solvePnP(objectPoints, corners, cameraMatrix, distCoeffs)
            #rvec, tvec = cv2.solvePnP(objectPoints, corners, cameraMatrix, None)
            # http://www.opencv.org.cn/opencvdoc/2.3.2/html/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
            # http://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
            #print cv2.Rodrigues(rvec)[0], linalg.norm(rvec), rvec
            #print tvec
            #cv2.composeRT
            #(cv2.findFundamentalMat, cv2.findHomography or from 'pose', cv2.estimateAffine3D)

            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            ## draw the rotated 3D object
            #imagePoints, jacobian = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)
            #for i in range(len(imagePoints)-1):
            #    cv2.line(im, tuple(imagePoints[i][0].astype(int)), tuple(imagePoints[i+1][0].astype(int)), (125.,125.,125.), 3)

            mat = np.eye(3)
            color = [(0., 0., 255.), (0., 255., 0.), (255., 0., 0.)]
            label = ['x', 'y', 'z']
            # axis-cross
            matD2raw, matD2norm, matnorm = self._util_getD2coords( mat, cameraMatrix, distCoeffs, sign=-1 )
            for i in range(3):
                imagePoints, D2norm, norm = matD2raw[:,:,:,i], 40*matD2norm[:,i], matnorm[:,i]
                #cv2.line(im, tuple(imagePoints[0][0].astype(int)), tuple(imagePoints[1][0].astype(int)), color[i], 1)
                #cv2.putText(im, label[i], tuple(imagePoints[1][0].astype(int)), cv2.FONT_HERSHEY_PLAIN, 1.5, color[i])
                cv2.line(im, (50,50), (50+D2norm[0].astype(int),50+D2norm[1].astype(int)), color[i], 1)
                cv2.putText(im, label[i], (50+D2norm[0].astype(int),50+D2norm[1].astype(int)), cv2.FONT_HERSHEY_PLAIN, 1., color[i])
            # rotated axis-cross
            matD2raw, matD2norm, matnorm = self._util_getD2coords( mat, cameraMatrix, distCoeffs, rvec=rvec, tvec=tvec )
            for i in range(3):
                imagePoints, D2norm, norm = matD2raw[:,:,:,i], 40*matD2norm[:,i], matnorm[:,i]
                cv2.line(im, tuple(imagePoints[0][0].astype(int)), tuple(imagePoints[1][0].astype(int)), color[i], 3)
                cv2.putText(im, label[i], tuple(imagePoints[1][0].astype(int)), cv2.FONT_HERSHEY_PLAIN, 1.5, color[i])
                cv2.line(im, (50,100), (50+D2norm[0].astype(int),100+D2norm[1].astype(int)), color[i], 1)
                cv2.putText(im, label[i], (50+D2norm[0].astype(int),100+D2norm[1].astype(int)), cv2.FONT_HERSHEY_PLAIN, 1., color[i])
                ortho = imagePoints[1][0]-imagePoints[0][0]     # z-axis is orthogonal to object surface
            ortho = ortho/linalg.norm(ortho)
# self-calculated rotated axis-cross
            rmat = np.zeros((3,4))
            rmat[:,0:3] = cv2.Rodrigues(rvec)[0]
            #rmat[:,3]   = tvec[:,0]
            mat = np.dot(rmat, cv2.convertPointsToHomogeneous(np.eye(3).astype('float32')).transpose()[:,0,:])
            ## rotation between z-axis (direction of view) and translation point tvec
            #vec = np.array([0.,0.,1.])
            #angle = np.arccos( np.dot(tvec[:,0], vec)/(linalg.norm(tvec[:,0])*linalg.norm(vec)) )
            #axis  = np.cross(tvec[:,0], vec)
            #rvec2 = axis/linalg.norm(axis) * -angle
            #rmat2 = cv2.Rodrigues(rvec2)[0]
            ##mat = np.dot(rmat2, mat)
            ##rot = cv2.Rodrigues(np.dot(rmat2, rmat[:,0:3]))[0]
            rot = rvec
            perp = mat
            # what follows SHOULD be invartiant of the choice of 'cameraMatrix' and 'distCoeffs' ... ! is it ?
            #cameraMatrix = np.eye(3)
            #distCoeffs = np.zeros((5,1))
            mat = np.dot((cameraMatrix), mat)       # linalg.inv(cameraMatrix)
            #_cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles = cv2.decomposeProjectionMatrix(rmat)
            #mat = np.dot(rotMatrix, np.eye(3))
            #matD2raw, matD2norm, matnorm = self._util_getD2coords( mat, cameraMatrix, distCoeffs )
            matD2raw, matD2norm, matnorm = self._util_getD2coords( mat, np.eye(3), distCoeffs )
            for i in range(3):
                imagePoints, D2norm, norm = matD2raw[:,:,:,i], 40*matD2norm[:,i], matnorm[:,i]
                D2norm = D2norm/linalg.norm(D2norm)*40
                cv2.line(im, (50,200), (50+D2norm[0].astype(int),200+D2norm[1].astype(int)), color[i], 1)
                cv2.putText(im, label[i], (50+D2norm[0].astype(int),200+D2norm[1].astype(int)), cv2.FONT_HERSHEY_PLAIN, 1., color[i])
                #ortho = imagePoints[1][0]-imagePoints[0][0]     # z-axis is orthogonal to object surface

            #cv2.imshow("win", im)
            #cv2.waitKey()
            pywikibot.output(u'result for calibrated camera:\n  rot=%s\n  perp=%s\n  perp2D=%s' % (rot.transpose()[0], perp[:,2], ortho))
            pywikibot.output(u'nice would be to do the same for uncalibrated/default cam settings')

            # still in testing phase; some of the values might have big errors
            self._info['Chessboard'][0]['Rotation']    = tuple(rot.transpose()[0])
            self._info['Chessboard'][0]['Perp_Dir']    = tuple(perp[:,2])
            self._info['Chessboard'][0]['Perp_Dir_2D'] = tuple(ortho)

        return

    def _util_getD2coords(self, D3coords, cameraMatrix, distCoeffs, rvec=None, tvec=None, sign=1):
        if rvec is None:
            rvec = np.zeros((3,1))
        if tvec is None:
            tvec = np.zeros((3,1))
        matD2raw  = np.zeros((2,1,2,D3coords.shape[0]))
        matD2norm = np.zeros((2,D3coords.shape[0]))
        matnorm   = np.zeros((1,D3coords.shape[0]))
        for i in range(D3coords.shape[0]):
#            D2raw, jacobian = cv2.projectPoints(np.array([[0.,0.,0.],D3coords[:,i]]), rvec, tvec, cameraMatrix, distCoeffs)
            D2raw, jacobian = cv2.projectPoints(np.array([[0.,0.,-1.],[D3coords[0,i],D3coords[1,i],D3coords[2,i]-1.]]), rvec, tvec, cameraMatrix, distCoeffs)
            D2norm = (D2raw[1][0]-D2raw[0][0])
            norm   = linalg.norm(D2norm)
#            D2norm[1] *= sign   # usual 2D coords <-> pixel/picture coords
            D2norm[0] *= sign   # usual 2D coords <-> pixel/picture coords
            D2norm    *= sign   # invert all
            matD2raw[:,:,:,i] = D2raw
            matD2norm[:,i]    = D2norm
            matnorm[:,i]      = norm
        matD2norm = matD2norm/max(matnorm[0])
        return (matD2raw, matD2norm, matnorm)

    def _util_get_DataTags_EXIF(self):
        # http://tilloy.net/dev/pyexiv2/tutorial.html
        # (is UNFORTUNATELY NOT ABLE to handle all tags, e.g. 'FacesDetected', ...)
        
        if hasattr(self, '_buffer_EXIF'):
            return self._buffer_EXIF

        res = {}
        enable_recovery()   # enable recovery from hard crash
        try:
            if hasattr(pyexiv2, 'ImageMetadata'):
                metadata = pyexiv2.ImageMetadata(self.image_path)
                metadata.read()
            
                for key in metadata.exif_keys:
                    res[key] = metadata[key]
                    
                for key in metadata.iptc_keys:
                    res[key] = metadata[key]
                    
                for key in metadata.xmp_keys:
                    res[key] = metadata[key]
            else:
                image = pyexiv2.Image(self.image_path)
                image.readMetadata()
            
                for key in image.exifKeys():
                    res[key] = image[key]
                    
                for key in image.iptcKeys():
                    res[key] = image[key]
                    
                #for key in image.xmpKeys():
                #    res[key] = image[key]
        except IOError:
            pass
        except RuntimeError:
            pass
        disable_recovery()  # disable since everything worked out fine
        
        
        # http://www.sno.phy.queensu.ca/~phil/exiftool/
        # MIGHT BE BETTER TO USE AS PYTHON MODULE; either by wrapper or perlmodule:
        # http://search.cpan.org/~gaas/pyperl-1.0/perlmodule.pod
        # (or use C++ with embbedded perl to write a python module)
        data = Popen("exiftool -j %s" % self.image_path, 
                     shell=True, stdout=PIPE).stdout.read()
        if not data:
            raise ImportError("exiftool not found!")
        try:   # work-a-round for badly encoded exif data (from pywikibot/comms/http.py)
            data = unicode(data, 'utf-8', errors = 'strict')
        except UnicodeDecodeError:
            data = unicode(data, 'utf-8', errors = 'replace')
        #res  = {}
        data = re.sub("(?<!\")\(Binary data (?P<size>\d*) bytes\)", "\"(Binary data \g<size> bytes)\"", data)  # work-a-round some issue
        for item in json.loads(data):
            res.update( item )
        #print res
        self._buffer_EXIF = res
        
        return self._buffer_EXIF

    def _detect_Faces_EXIF(self):
        res = self._util_get_DataTags_EXIF()
        
        # http://u88.n24.queensu.ca/exiftool/forum/index.php?topic=3156.0
        # http://u88.n24.queensu.ca/pub/facetest.pl
        # ( all scaling stuff ignored (!) and some strongly simplified (!) )
        # Example: 'File:Annagrah-2 041.JPG' (canon)
        if 'Make' in res:
            make = res['Make'].lower()
        else:
            make = ''
        found = set(res.keys())
        data  = []

        if 'ImageWidth' in res:
            (width, height) = (str(res['ImageWidth']), str(res['ImageHeight']))
            (width, height) = (re.sub(u'p[tx]', u'', width), re.sub(u'p[tx]', u'', height))
            try:
                (width, height) = (int(float(width)+0.5), int(float(height)+0.5))
            except ValueError:
                pywikibot.output(u'WARNING: %s contains incompatible unit(s), skipped' % ((width, height),))
                return
        else:
            (width, height) = self.image_size
        wasRotated = (height > width)
        
        if   True in [item in make for item in ['sony', 'nikon', 'panasonic', 'casio', 'ricoh']]:
            # UNTESTED: ['sony', 'nikon', 'casio', 'ricoh']
            #   TESTED: ['panasonic']
            if set(['FacesDetected', 'Face1Position']).issubset(found):
                i = 1
                if 'FaceOrientation' in res:
                    pywikibot.output(res['FaceOrientation'])    # for rotation 'rot'
                # 'crop' for 'casio' omitted here...
                aspect = float(height)/width
                if (aspect <= 3./4):
                    (fw, fh) = (320, 320 * aspect)
                else:
                    (fw, fh) = (240 / aspect, 240)
                #(sx, sy) = (1./width, 1./height)
                (sx, sy) = (1./fw, 1./fh)
                if 'FaceDetectFrameSize' in res:
                    (width, height) = map(int, res['FaceDetectFrameSize'].split(' '))
                    (sx, sy) = (1./width, 1./height)
                while (('Face%iPosition'%i) in res) and (i <= int(res['FacesDetected'])):
                    buf = map(int, res['Face%iPosition'%i].split(' '))
                    (x1, y1) = ((buf[0]-buf[2]/2)*sx, (buf[1]-buf[3]/2)*sy)    # 'panasonic'
                    (x2, y2) = (x1+buf[2]*sx, y1+buf[3]*sy)                    #
                    #(x1, y1) = (buf[1]*sx, buf[0]*sy)
                    #(x2, y2) = (x1+buf[3]*sx, y1+buf[2]*sy)
                    data.append({ 'Position': (x1, y1, x2, y2) })
                    if ('RecognizedFace%iName'%i) in res:
                        pywikibot.output(str((res['RecognizedFace%iName'%i], res['RecognizedFace%iAge'%i])))
                    i += 1
        elif 'fujifilm' in make:
            # UNTESTED: 'fujifilm'
            if set(['FacesDetected', 'FacePositions']).issubset(found):
                buf = map(int, res['FacePositions'].split(' '))
                (sx, sy) = (1./width, 1./height)
                for i in range(int(res['FacesDetected'])):
                    data.append({ 'Position': [buf[i*4]*sx,   buf[i*4+1]*sy, 
                                               buf[i*4+2]*sx, buf[i*4+3]*sy] })
                    if ('Face%iName'%i) in res:
                        pywikibot.output(str((res['Face%iName'%i], res['Face%iCategory'%i], res['Face%iBirthday'%i])))
        elif 'olympus' in make:
            # UNTESTED: 'olympus'
            if set(['FacesDetected', 'FaceDetectArea']).issubset(found):
                buf = map(int, res['FacesDetected'].split(' '))
                if buf[0] or buf[1]:
                    buf = map(int, res['FaceDetectArea'].split(' '))
                    for i in range(int(res['MaxFaces'])):
                        data.append({ 'Position': [buf[i*4], buf[i*4+1], buf[i*4+2], buf[i*4+3]] })
        elif True in [item in make for item in ['pentax', 'sanyo']]:
            # UNTESTED: ['pentax', 'sanyo']
            if set(['FacesDetected']).issubset(found):
                i = 1
                (sx, sy) = (1./width, 1./height)
                while ('Face%iPosition'%i) in res:
                    buf = map(int, res['Face%iPosition'%i].split(' ') + \
                                   res['Face%iSize'%i].split(' '))
                    (x1, y1) = ((buf[0] - buf[2]/2.)*sx, (buf[1] - buf[3]/2.)*sy)
                    (x2, y2) = (x1+buf[2]*sx, y1+buf[3]*sy)
                    data.append({ 'Position': (x1, y1, x2, y2) })
                    i += 1
                if 'FacePosition' in res:
                    buf = map(int, res['FacePosition'].split(' ') + ['100', '100']) # how big is the face?
                    (x1, y1) = (buf[0]*sx, buf[1]*sy)
                    (x2, y2) = (buf[2]*sx, buf[3]*sy)
                    data.append({ 'Position': (x1, y1, x2, y2) })
        elif 'canon' in make:
            if   set(['FacesDetected', 'FaceDetectFrameSize']).issubset(found) \
                 and (int(res['FacesDetected'])):
                # TESTED: older models store face detect information
                (width, height) = map(int, res['FaceDetectFrameSize'].split(' '))   # default: (320,240)
                (sx, sy) = (1./width, 1./height)
                fw = res['FaceWidth'] or 35
                i = 1
                while ('Face%iPosition'%i) in res:
                    buf = map(int, res['Face%iPosition'%i].split(' '))
                    (x1, y1) = ((buf[0] + width/2. - fw)*sx, (buf[1] + height/2. - fw)*sy)
                    (x2, y2) = (x1 + fw*2*sx, y1 + fw*2*sy)
                    data.append({ 'Position': (x1, y1, x2, y2) })
                    i += 1
            elif set(['ValidAFPoints', 'AFImageWidth', 'AFImageHeight',
                      'AFAreaXPositions', 'AFAreaYPositions', 'PrimaryAFPoint']).issubset(found):
                # TESTED: newer models use AF points
                (width, height) = (int(res['AFImageWidth']), int(res['AFImageHeight']))
                if ('AFAreaMode' in res) and ('Face' in res['AFAreaMode']):
                    buf_x = res['AFAreaXPositions'].split(' ')
                    buf_y = res['AFAreaYPositions'].split(' ')
                    buf_w = buf_h = [100] * len(buf_x) # how big is the face? (else)
                    if   'AFAreaWidths' in res:
                        buf_w = map(int, res['AFAreaWidths'].split(' '))
                        buf_h = map(int, res['AFAreaHeights'].split(' '))
                    elif 'AFAreaWidth' in res:
                        buf_w = [int(res['AFAreaWidth'])]  * len(buf_x)
                        buf_h = [int(res['AFAreaHeight'])] * len(buf_x)
                    else:
                        pywikibot.output(u'No AF area size')
                    # conversion to positive coordinates
                    buf_x = [ int(x) + width/2. for x in buf_x ]
                    buf_y = [ int(y) + height/2. for y in buf_y ]
                    # EOS models have Y flipped
                    if ('Model' in res) and ('EOS' in res['Model']):
                        buf_y = [ height - y for y in buf_y ]
                    (sx, sy) = (1./width, 1./height)
                    for i in range(int(res['ValidAFPoints'])):
                        (x1, y1) = ((buf_x[i]-buf_w[i]/2)*sx, (buf_y[i]-buf_h[i]/2)*sy)
                        (x2, y2) = (x1+buf_w[i]*sx, y1+buf_h[i]*sy)
                        data.append({ 'Position': (x1, y1, x2, y2) })
        else:
            # not supported (yet...)
            available = [item in res for item in ['FacesDetected', 'ValidAFPoints']]
            unknown   = ['face' in item.lower() for item in res.keys()]
            if make and (True in (available+unknown)):
                pywikibot.output(u"WARNING: skipped '%s' since not supported (yet) [_detect_Faces_EXIF]" % make)
                pywikibot.output(u"WARNING: FacesDetected: %s - ValidAFPoints: %s" % tuple(available))
        
        # finally, rotate face coordinates if image was rotated
        if wasRotated:
            rot = 270
            # variable rotation omitted here... ($$faceInfo{Rotation})

        for i, d in enumerate(data):
            # rotate face coordinates
            p = data[i]['Position']
            if wasRotated:
                if (rot == 90):
                    p = (p[1], 1-p[0], p[3], 1-p[2])
                else:
                    p = (1-p[1], p[0], 1-p[3], p[2])
                if 'Rotation' in data[i]:
                    data[i]['Rotation'] -= rot
                    data[i]['Rotation'] += 360 if data[i]['Rotation'] < 0 else 0

            # rescale relative sizes to real pixel values
            p = (p[0]*self.image_size[0] + 0.5, p[1]*self.image_size[1] + 0.5, 
                 p[2]*self.image_size[0] + 0.5, p[3]*self.image_size[1] + 0.5)
            # change from (x1, y1, x2, y2) to (x, y, w, h)
            #data[i]['Position'] = (p[0], p[1], p[0]-p[2], p[3]-p[1])
            data[i]['Position'] = (min(p[0],p[2]), min(p[1],p[3]), 
                                   abs(p[0]-p[2]), abs(p[3]-p[1]))

            data[i] = { 'Position': tuple(map(int, data[i]['Position'])),
                        'ID':       (i+1),
                        'Type':     u'Exif',
                        'Eyes':     [],
                        'Mouth':    (),
                        'Nose':     (), }
            data[i]['Coverage'] = float(data[i]['Position'][2]*data[i]['Position'][3])/(self.image_size[0]*self.image_size[1])

        # (exclusion of duplicates is done later by '_util_merge_Regions')

        self._info['Faces'] += data
        return
    
    def _detect_History_EXIF(self):
        self._info['History'] = []

        res = self._util_get_DataTags_EXIF()

        #a = []
        #for k in res.keys():
        #    if 'history' in k.lower():
        #        a.append( k )
        #for item in sorted(a):
        #    print item
        # http://tilloy.net/dev/pyexiv2/api.html#pyexiv2.xmp.XmpTag
        #print [getattr(res['Xmp.xmpMM.History'], item) for item in ['key', 'type', 'name', 'title', 'description', 'raw_value', 'value', ]]
        result = []
        i = 1
        while (('Xmp.xmpMM.History[%i]' % i) in res):
            data = { 'ID':        i,
                     'Software':  u'-',
                     'Timestamp': u'-',
                     'Action':    u'-',
                     'Info':      u'-', }
            if   ('Xmp.xmpMM.History[%i]/stEvt:softwareAgent'%i) in res:
                data['Software']  = res['Xmp.xmpMM.History[%i]/stEvt:softwareAgent'%i].value
                data['Timestamp'] = res['Xmp.xmpMM.History[%i]/stEvt:when'%i].value
                data['Action']    = res['Xmp.xmpMM.History[%i]/stEvt:action'%i].value
                if ('Xmp.xmpMM.History[%i]/stEvt:changed'%i) in res:
                    data['Info']  = res['Xmp.xmpMM.History[%i]/stEvt:changed'%i].value
                #print res['Xmp.xmpMM.History[%i]/stEvt:instanceID'%i].value
                result.append( data )
            elif ('Xmp.xmpMM.History[%i]/stEvt:parameters'%i) in res:
                data['Action']    = res['Xmp.xmpMM.History[%i]/stEvt:action'%i].value
                data['Info']      = res['Xmp.xmpMM.History[%i]/stEvt:parameters'%i].value
                #data['Action']    = data['Info'].split(' ')[0]
                result.append( data )
            else:
                pass
            i += 1
        
        self._info['History'] = result
        return

    def _util_get_DataStreams_FFMPEG(self):
        if hasattr(self, '_buffer_FFMPEG'):
            return self._buffer_FFMPEG

        # (similar as in '_util_get_DataTags_EXIF')
# switch this part off since 'ffprobe' (on toolserver) is too old; TS-1449
#        data = Popen("ffprobe -v quiet -print_format json -show_format -show_streams %s" % self.image_path, 
        proc = Popen("ffprobe -v quiet -show_format -show_streams %s" % self.image_path,#.replace('%', '%%'), 
                     shell=True, stdout=PIPE)#.stdout.read()
        proc.wait()
        if proc.returncode == 127:
            raise ImportError("ffprobe (ffmpeg) not found!")
        data = proc.stdout.read().strip()
#        self._buffer_FFMPEG = json.loads(data)
        res, key, cur = {}, '', {}
        for item in data.splitlines():
            if (item[0] == '['):
                if not (item[1] == '/'):
                    key = item[1:-1]
                    cur = {}
                    if key not in res:
                        res[key] = []
                else:
                    res[key].append( cur )
            else:
                val = item.split('=')
                cur[val[0].strip()] = val[1].strip()
        if res:
            res = { 'streams': res['STREAM'], 'format': res['FORMAT'][0] }
        self._buffer_FFMPEG = res
        
        return self._buffer_FFMPEG

    def _detect_Streams_FFMPEG(self):
        self._info['Streams'] = []

        # skip file formats that interfere and can cause strange results (pdf is oga?!)
        # or file formats not supported (yet?)
        if (self.image_mime[1] in ['pdf']) or (self.image_fileext in [u'.svg']):
            return

        # audio and video streams files (ogv, oga, ...)
        d = self._util_get_DataStreams_FFMPEG()
        if not d:
            return

        result = []
        for s in d['streams']:
            #print s
            if   (s["codec_type"] == "video"):
                rate = s["avg_frame_rate"]
                dim = (int(s["width"]), int(s["height"]))
                #asp  = s["display_aspect_ratio"]
            elif (s["codec_type"] == "audio"):
# switch this part off since 'ffprobe' (on toolserver) is too old
#                rate = u'%s/%s/%s' % (s["channels"], s["sample_fmt"], s["sample_rate"])
                rate = u'%s/%s/%s' % (s["channels"], u'-', int(float(s["sample_rate"])))
                dim  = None
            elif (s["codec_type"] == "data"):
                rate = None
                dim  = None

            result.append({ 'ID':         int(s["index"]) + 1,
                            'Format':     u'%s/%s' % (s["codec_type"], s.get("codec_name",u'?')),
                            'Rate':       rate or u'-',
                            'Dimensions': dim or (None, None),
                            })

        if 'image' not in d["format"]["format_name"]:
            self._info['Streams'] = result
        return

    def _util_merge_Regions(self, regs, sub=False, overlap=False, close=False):
        # sub=False, overlap=False, close=False ; level 0 ; similar regions, similar position (default)
        # sub=True,  overlap=False, close=False ; level 1 ; region contained in other, any shape/size
        # sub=False, overlap=True,  close=False ; level 2 ; center of region conatained in other
        # sub=False, overlap=False, close=True  ; level 3 ; regions placed close together

        if not regs:
            return ([], [])

        dmax = np.linalg.norm(self.image_size)
        #thsr = 1.0      # strict: if it is contained completely
        thsr = 0.95      # a little bit tolerant: nearly completly contained (or 0.9)
        drop = []
        for i1, r1i in enumerate(regs):
            r1 = np.float_(r1i)
            (xy1, wh1) = (r1[0:2], r1[2:4])
            c1 = xy1 + wh1/2
            a1 = wh1[0]*wh1[1]

            # check for duplicates (e.g. similar regions in similar position)
            i2 = 0
            while (i2 < i1):
                r2i, r2 = regs[i2], np.float_(regs[i2])
                (xy2, wh2) = (r2[0:2], r2[2:4])
                c2 = xy2 + wh2/2
                a2 = wh2[0]*wh2[1]

                dr = np.linalg.norm(c1-c2)/dmax
                intersect = gtk.gdk.Rectangle(*r1i).intersect(gtk.gdk.Rectangle(*r2i))
                area = intersect.width*intersect.height
                ar1, ar2 = area/a1, area/a2
                check = [(1-dr), ar1, ar2]
                # (I assume the 1. condition (1-dr) to be always true if the 2.
                # and 3. are - so it's obsolete... how is the analytic relation?)

                # add the first match (first is assumed to be the best one) / drop second one
                #print check, np.average(check), np.std(check)
                if (np.average(check) >= 0.9) and (np.std(check) <= 0.1):
                #if (np.average(check) >= 0.85) and (np.std(check) <= 0.1):
                    drop.append( i1 )
                # remove all sub-rect/-regions (all regions fully contained in other)
                if sub:
                    #drop.append( [i1, i2][check[0:2].index(1.0)] )
                    if   (ar1 >= thsr) and (i2 not in drop):
                        drop.append( i1 )
                    elif (ar2 >= thsr) and (i1 not in drop):
                        drop.append( i2 )
                # from '_detect_Faces_CV()'
                if overlap:
                    if (r2[0] <= c1[0] <= (r2[0] + r2[2])) and \
                       (r2[1] <= c1[1] <= (r2[1] + r2[3])) and (i2 not in drop):
                        drop.append( i1 )
                if close:
                    if (check[0] >= 0.985) and (i2 not in drop):     # at least (!)
                        drop.append( i1 )

                i2 += 1

        drop = sorted(list(set(drop)))
        drop.reverse()
        for i in drop:
            del regs[i]

        return (regs, drop)

    def _detect_AudioFeatures_YAAFE(self):
        # http://yaafe.sourceforge.net/manual/tools.html
        # http://yaafe.sourceforge.net/manual/quickstart.html - yaafe.py
        # ( help: yaafe.py -h / features: yaafe.py -l )
        #
        # compile yaafe on fedora:
        # 1.) get and compile 'argtable2' (2-13)
        #     1.1 download from http://argtable.sourceforge.net/
        #     1.2 unpack and cd to directory
        #     1.3 $ ccmake .
        #     1.4 set: CMAKE_BUILD_TYPE = Release
        #     1.5 press: c, g (in order to configure and generate)
        #     1.6 $ make
        # 2.) get and compile 'yaafe'
        #     1.1 download from http://yaafe.sourceforge.net/
        #     1.2 unpack and cd to directory
        #     1.3 $ ccmake .
        #     1.4 set: ARGTABLE2_INCLUDE_DIR = /home/ursin/Desktop/argtable2-13/src
        #              ARGTABLE2_LIBRARY     = /home/ursin/Desktop/argtable2-13/src/libargtable2.a
        #              ...
        #              DL_INCLUDE_DIR        = /usr/include
        #              DL_LIBRARY            = /usr/lib64/libdl.so
        #              FFTW3_INCLUDE_DIR     = /usr/include
        #              FFTW3_LIBRARY         = /usr/lib64/libfftw3.so
        #              HDF5_HL_LIBRARY       = /usr/lib64/libhdf5_hl.so
        #              HDF5_INCLUDE_DIR      = /usr/include
        #              HDF5_LIBRARY          = /usr/lib64/libhdf5.so
        #              LAPACK_LIBRARY        = /usr/lib64/liblapack.so
        #              MATLAB_ROOT           = MATLAB_ROOT-NOTFOUND
        #              MPG123_INCLUDE_DIR    = /usr/include
        #              MPG123_LIBRARY        = /usr/lib64/libmpg123.so
        #              RT_LIBRARY            = /usr/lib64/librt.so
        #              SNDFILE_INCLUDE_DIR   = /usr/include
        #              SNDFILE_LIBRARY       = /usr/lib64/libsndfile.so
        #              ...
        #              WITH_FFTW3            = ON
        #              WITH_HDF5             = ON
        #              WITH_LAPACK           = ON
        #              WITH_MATLAB_MEX       = OFF
        #              WITH_MPG123           = ON
        #              WITH_SNDFILE          = ON
        #              WITH_TIMERS           = ON
        #              (use t to toggle to more advanced options)
        #              CMAKE_CXX_FLAGS       = -fpermissive
        #              CMAKE_C_FLAGS         = -fpermissive
        #         (install all needed dependencies/packages into the OS also)
        #     1.5 press: c, g (in order to configure and generate)
        #     1.6 $ make
        #     1.7 $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ursin/Desktop/yaafe-v0.64/src_cpp/yaafe-python/:/home/ursin/Desktop/yaafe-v0.64/src_cpp/yaafe-io/:/home/ursin/Desktop/yaafe-v0.64/src_cpp/yaafe-core/:/home/ursin/Desktop/yaafe-v0.64/src_cpp/yaafe-components/
        #         $ export YAAFE_PATH=/home/ursin/Desktop/yaafe-v0.64/src_python/
        #         $ export PYTHONPATH=/home/ursin/Desktop/yaafe-v0.64/src_python

        # skip file formats not supported (yet?)
        if (self.image_mime[1] in ['ogg']):
            return

        self._info['Audio'] = []

        import yaafelib as yaafe

        # use WAV, OGG, MP3 (and others) audio file formats
        #audiofile = '/home/ursin/data/09Audio_UNS/Amy MacDonald - This Is The Life (2007) - Pop/01-amy_macdonald-mr_rock_and_roll.mp3'
        audiofile = self.image_path

        yaafe.setVerbose(True)
        #print 'Yaafe v%s'%yaafe.getYaafeVersion()

        # Load important components
        if (yaafe.loadComponentLibrary('yaafe-io')!=0):
            pywikibot.output(u'WARNING: cannot load yaafe-io component library !')   # ! needed, else it will crash !

        # Build a DataFlow object using FeaturePlan
        fp = yaafe.FeaturePlan(sample_rate=44100, normalize=0.98, resample=False)
        #fp.addFeature('am: AmplitudeModulation blockSize=512 stepSize=256')
        #fp.addFeature('ac: AutoCorrelation blockSize=512 stepSize=256')
        #fp.addFeature('cdod: ComplexDomainOnsetDetection blockSize=512 stepSize=256')
        #fp.addFeature('erg: Energy blockSize=512 stepSize=256')
        #fp.addFeature('e: Envelope blockSize=512 stepSize=256')
        fp.addFeature('ess: EnvelopeShapeStatistics blockSize=512 stepSize=256')
        #fp.addFeature('f: Frames blockSize=512 stepSize=256')
        #fp.addFeature('lpc: LPC blockSize=512 stepSize=256')
        #fp.addFeature('lsf: LSF blockSize=512 stepSize=256')
        #fp.addFeature('l: Loudness blockSize=512 stepSize=256')
        #fp.addFeature('mfcc: MFCC blockSize=512 stepSize=256')
        ## features: AutoCorrelationPeaksIntegrator, Cepstrum, Derivate, HistogramIntegrator, SlopeIntegrator, StatisticalIntegrator
        #fp.addFeature('mfcc_d1: MFCC blockSize=512 stepSize=256 > Derivate DOrder=1')
        #fp.addFeature('mfcc_d2: MFCC blockSize=512 stepSize=256 > Derivate DOrder=2')
        #fp.addFeature('mas: MagnitudeSpectrum blockSize=512 stepSize=256')
        #fp.addFeature('mes: MelSpectrum blockSize=512 stepSize=256')
        #fp.addFeature('obsi: OBSI blockSize=512 stepSize=256')
        #fp.addFeature('obsir: OBSIR blockSize=512 stepSize=256')
        #fp.addFeature('psh: PerceptualSharpness blockSize=512 stepSize=256')
        #fp.addFeature('psp: PerceptualSpread blockSize=512 stepSize=256')
        #fp.addFeature('scfpb: SpectralCrestFactorPerBand blockSize=512 stepSize=256')
        #fp.addFeature('sd: SpectralDecrease blockSize=512 stepSize=256')
        #fp.addFeature('sfa: SpectralFlatness blockSize=512 stepSize=256')
        #fp.addFeature('sfpb: SpectralFlatnessPerBand blockSize=512 stepSize=256')
        #fp.addFeature('sfu: SpectralFlux blockSize=512 stepSize=256')
        #fp.addFeature('sr: SpectralRolloff blockSize=512 stepSize=256')
        fp.addFeature('sss: SpectralShapeStatistics blockSize=512 stepSize=256')
        #fp.addFeature('ss: SpectralSlope blockSize=512 stepSize=256')
        #fp.addFeature('sv: SpectralVariation blockSize=512 stepSize=256')
        fp.addFeature('tss: TemporalShapeStatistics blockSize=512 stepSize=256')
        fp.addFeature('zcr: ZCR blockSize=512 stepSize=256')
        df = fp.getDataFlow()

        ## or load a DataFlow from dataflow file.
        #df = DataFlow()
        #df.load(dataflow_file)

        #fp.getDataFlow().save('')
        #print df.display()

        # configure an Engine
        engine = yaafe.Engine()
        engine.load(df)
        # extract features from an audio file using AudioFileProcessor
        afp = yaafe.AudioFileProcessor()
        #afp.setOutputFormat('csv','',{})       # ! needed, else it will crash ! (but now produces file output)
        #afp.processFile(engine,audiofile)
        #feats = engine.readAllOutputs()
        ## and play with your features
        #print feats

        # extract features from an audio file and write results to csv files
        afp.setOutputFormat('csv','output',{'Precision':'8'})
        afp.processFile(engine,audiofile)
        # this creates output/myaudio.wav.mfcc.csv, .mfcc_d1.csv and .mfcc_d2.csv files.

        ## extract features from a numpy array
        #audio = np.random.randn(1,100000)
        #feats = engine.processAudio(audio)
        ## and play with your features
        #print feats

        import csv
        data = {}
        for ext in ['ess', 'sss', 'tss', 'zcr']:
            fn = 'output' + audiofile + ('.%s.csv' % ext)
            with open(fn, 'rb') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                d = [row for row in reader]
                d = np.array(d[5:])    # cut header and convert to numpy
                d = np.float_(d)
                d = tuple(np.average(d, axis=0))
                pywikibot.output(ext)
                #if ext in ['ess', 'sss', 'tss']:
                #    pywikibot.output(u"centroid: %s\nspread: %s\nskewness: %s\nkurtosis: %s\n" % d)
                #elif ext in ['zcr']:
                #    pywikibot.output(u"zero-crossing rate: %s\n" % d)
                data[ext.upper()] = d
            os.remove(fn)
            # remove folder too...

        self._info['Audio'] = [data]
        return


# all classification and categorization methods and definitions - default variation
#  use simplest classification I can think of (self-made) and do categorization
#  mostly based on filtered/reported features
class CatImages_Default(FileData):
    #ignore = []
    ignore = ['color']
    
    _thrhld_group_size = 4
    #_thrshld_guesses = 0.1
    _thrshld_default = 0.75

    # for '_detect_Trained_CV'
    cascade_files = [(u'Legs', 'haarcascade_lowerbody.xml'),
                     (u'Torsos', 'haarcascade_upperbody.xml'),
                     (u'Ears', 'haarcascade_mcs_leftear.xml'),
                     (u'Ears', 'haarcascade_mcs_rightear.xml'),
                     (u'Eyes', 'haarcascade_lefteye_2splits.xml'),        # (http://yushiqi.cn/research/eyedetection)
                     (u'Eyes', 'haarcascade_righteye_2splits.xml'),       # (http://yushiqi.cn/research/eyedetection)
                     #dtbext/opencv/haarcascades/haarcascade_mcs_lefteye.xml
                     #dtbext/opencv/haarcascades/haarcascade_mcs_righteye.xml
                     # (others include indifferent (left and/or right) and pair)
                     (u'Automobiles', 'cars3.xml'),                       # http://www.youtube.com/watch?v=c4LobbqeKZc
                     (u'Hands', '1256617233-2-haarcascade-hand.xml', 300.),]    # http://www.andol.info/
                     # ('Hands' does not behave very well, in fact it detects any kind of skin and other things...)
                     #(u'Aeroplanes', 'haarcascade_aeroplane.xml'),]      # e.g. for 'Category:Unidentified aircraft'

    # very simple / rought / poor-man's min. thresshold classification
    # (done by guessing, does not need to be trained)
    # replace/improve this with RTrees, KNearest, Boost, SVM, MLP, NBayes, ...
    def classifyFeatures(self):
        # classification of detected features (should use RTrees, KNearest, Boost, SVM, MLP, NBayes, ...)
        # ??? (may be do this in '_cat_...()' or '_filter_...()' ?!?...)

        # Faces and eyes (opencv pre-trained haar and extracted EXIF data)
        for i in range(len(self._info['Faces'])):
            if self._info['Faces'][i]['Type'] == u'Exif':
                c = self._thrshld_default
            else:
                c = (len(self._info['Faces'][i]['Eyes']) + 2.) / 4.
            self._info['Faces'][i]['Confidence'] = c
            self._info['Faces'][i]['ID'] = i+1

        # Segments and colors / Average color
        #max_dim = max(self.image_size)
        for i in range(len(self._info['ColorRegions'])):
            data = self._info['ColorRegions'][i]

            # has to be in descending order since only 1 resolves (!)
            #if   (data['Coverage'] >= 0.40) and (data['Delta_E']  <=  5.0):
            #    c = 1.0
            ##elif (data['Coverage'] >= 0.20) and (data['Delta_E']  <= 15.0):
            ##elif (data['Coverage'] >= 0.20) and (data['Delta_E']  <= 10.0):
            #elif (data['Coverage'] >= 0.25) and (data['Delta_E']  <= 10.0):
            #    c = 0.75
            #elif (data['Coverage'] >= 0.10) and (data['Delta_E']  <= 20.0):
            #    c = 0.5
            #else:
            #    c = 0.1
            ca = (data['Coverage'])**(1./7)                 # 0.15 -> ~0.75
            #ca = (data['Coverage'])**(1./6)                 # 0.20 -> ~0.75
            #ca = (data['Coverage'])**(1./5)                 # 0.25 -> ~0.75
            #ca = (data['Coverage'])**(1./4)                 # 0.35 -> ~0.75
            ##cb = (0.02 * (50. - data['Delta_E']))**(1.2)    # 10.0 -> ~0.75
            #cb = (0.02 * (50. - data['Delta_E']))**(1./2)   # 20.0 -> ~0.75
            ##cb = (0.02 * (50. - data['Delta_E']))**(1./3)   # 25.0 -> ~0.75
            #cc = (1. - (data['Delta_R']/max_dim))**(1.)     # 0.25 -> ~0.75
            #c  = ( 3*ca + cb ) / 4
            #c  = ( cc + 6*ca + 2*cb ) / 9
            c  = ca
            self._info['ColorRegions'][i]['Confidence'] = c

        # People/Pedestrian (opencv pre-trained hog and haarcascade)
        for i in range(len(self._info['People'])):
            data = self._info['People'][i]

            if (data['Coverage'] >= 0.20):
                c = 0.75
            if (data['Coverage'] >= 0.10):      # at least 10% coverage needed
                c = 0.5
            else:
                c = 0.1
            self._info['People'][i]['Confidence'] = c

        # general (opencv pre-trained, third-party and self-trained haar
        # and cascade) classification
        for cf in self.cascade_files:
            cat = cf[0]
            for i in range(len(self._info[cat])):
                data = self._info[cat][i]
                # detect images with this as one of the main contents only thus
                # high coverage requested as a minimal confidence estimation
                self._info[cat][i]['Confidence'] = (data['Coverage'])**(1./5)  # 0.25 -> ~0.75

        # barcode and Data Matrix recognition (libdmtx/pydmtx, zbar, gocr?)
        for i in range(len(self._info['OpticalCodes'])):
            self._info['OpticalCodes'][i]['Confidence'] = min(0.75*self._info['OpticalCodes'][i]['Quality']/10., 1.)

        # Chessboard (opencv reference detector)
        for i in range(len(self._info['Chessboard'])):
            self._info['Chessboard'][i]['Confidence'] = len(self._info['Chessboard'][i]['Corners'])/49.

        ## Geometric object (opencv hough line, circle, edges, corner, ...)
        #if self._info['Geometry']:
        #    self._info['Geometry'][0]['Confidence'] = 1. - self._info['Geometry'][0]['Edge_Ratio']

    # Category:Unidentified people
    def _cat_people_People(self):
        #relevance = bool(self._info_filter['People'])
        relevance = self._cat_people_Groups()[1]

        return (u'Unidentified people', relevance)

    # Category:Unidentified people
    #def _cat_multi_People(self):
    def _cat_face_People(self):
        relevance = bool(self._info_filter['Faces'])
        #relevance = bool(self._info_filter['People']) or relevance

        return (u'Unidentified people', relevance)

    # Category:Groups
    def _cat_people_Groups(self):
        result = self._info_filter['People']

        relevance = (len(result) >= self._thrhld_group_size) and \
                    (not self._cat_coloraverage_Graphics()[1])

        return (u'Groups', relevance)

    # Category:Groups
    def _cat_face_Groups(self):
        result = self._info_filter['Faces']

        #if not (len(result) > 1): # 5 should give 0.75 and get reported
        #    relevance = 0.
        #else:
        #    relevance = 1 - 1./(len(result)-1)
        relevance = (len(result) >= self._thrhld_group_size)

        return (u'Groups', relevance)

    # Category:Faces
    def _cat_face_Faces(self):
        result = self._info_filter['Faces']

        #return (u'Faces', ((len(result) == 1) and (result[0]['Coverage'] >= .50)))
        return (u'Faces', ((len(result) == 1) and (result[0]['Coverage'] >= .40)))

    # Category:Portraits
    def _cat_face_Portraits(self):
        result = self._info_filter['Faces']

        #return (u'Portraits', ((len(result) == 1) and (result[0]['Coverage'] >= .25)))
        return (u'Portraits', ((len(result) == 1) and (result[0]['Coverage'] >= .20)))

    # Category:Barcode
    def _cat_code_Barcode(self):
        relevance = bool(self._info_filter['OpticalCodes'])

        return (u'Barcode', relevance)

    # Category:Chessboards
    def _cat_chess_Chessboards(self):
        relevance = bool(self._info_filter['Chessboard'])

        return (u'Chessboards', relevance)

    # Category:Books (literature) in PDF
    def _cat_text_BooksPDF(self):
        pdf    = u'PDF' in self._info_filter['Properties'][0]['Format']
        result = self._info_filter['Text']
        relevance = pdf and len(result) and \
                    (self._info_filter['Properties'][0]['Pages'] >= 10) and \
                    (result[0]['Size'] >= 5E4) and (result[0]['Lines'] >= 1000)

        return (u'Books (literature) in PDF', relevance)

    # Category:Animated GIF
    # Category:Animated PNG‎
    # (Category:Animated SVG‎)
    def _cat_prop_Animated_general(self):
        result = self._info_filter['Properties']
        relevance = result and (result[0]['Pages'] > 1) and \
                    (result[0]['Format'] in [u'GIF', u'PNG'])

        return (u'Animated %s' % result[0]['Format'], relevance)

    # Category:Human ears
    def _cat_ears_HumanEars(self):
        relevance = bool(self._info_filter['Ears'])

        return (u'Human ears', relevance)

    # Category:Human eyes
    def _cat_eyes_HumanEyes(self):
        relevance = bool(self._info_filter['Eyes'])

        return (u'Human eyes', relevance)

    # Category:Ogg sound files
    def _cat_streams_OggSoundFiles(self):
        result = self._info_filter['Streams']

        return (u'Ogg sound files', ((len(result) == 1) and (u'audio/' in result[0]['Format'])))

    # Category:Videos
    def _cat_streams_Videos(self):
        result = self._info_filter['Streams']

        return (u'Videos', (True in [u'video/' in s['Format'] for s in result]))

    # Category:Graphics
    def _cat_coloraverage_Graphics(self):
        result = self._info_filter['ColorAverage']
        relevance = (result and result[0]['Gradient'] < 0.1) and \
                    (0.005 < result[0]['Peaks'] < 0.1)  # black/white texts are below that
                    #(result[0]['FFT_Peaks'] < 500)      # has to be tested first !!!

        return (u'Graphics', bool(relevance))

    # Category:Categorized by DrTrigonBot
    def _addcat_BOT(self):
        # - ALWAYS -
        return (u"Categorized by DrTrigonBot", True)

    # (Category:BMP)
    # (Category:PNG)
    # (Category:JPEG)
    # Category:TIFF files
    # (may be more image formats/extensions according to PIL, e.g. SVG, ...)
    # Category:PDF files
    def _addcat_prop_general(self):
        fmt = self._info_filter['Properties'][0]['Format']
        if   u'TIFF' in fmt:
            fmt = u'TIFF images'
        #elif u'SVG' in fmt:
        #    # additional to PIL (rsvg, ...)
        #    # should be added as template instead of category (!)
        #    fmt = u''
        elif u'PDF' in fmt:
            # additional to PIL (...)
            fmt = u'PDF files'
        else:
            # disable ALL categorization, except the listed exceptions above
            # (BMP, PNG, JPEG, OGG; no general catgeory available, ...)
            fmt = u''
        # PIL: http://www.pythonware.com/library/pil/handbook/index.htm

        return (fmt, bool(fmt))

#    # TODO: add templates (conditional/additional like 'addcat')
#    # Category:SVG - Category:Valid SVG‎ - Category:Invalid SVG
#    # {{ValidSVG}} - {{InvalidSVG}}
#    def _addtempl_prop_SVN(self):
#        fmt = self._info_filter['Properties'][0]['Format']
#        d   = { u'Valid SVG':   u'{{ValidSVG}}',
#                u'Invalid SVG': u'{{InvalidSVG}}', }
#        fmt = d.get(fmt, u'')
#
#        return (fmt, bool(fmt))

#    # Category:Unidentified people
#    def _guess_Classify_People(self):
#        pass
#    # Category:Unidentified maps
#    def _guess_Classify_Maps(self):
#        pass
#    # Category:Unidentified flags
#    def _guess_Classify_Flags(self):
#        pass
#    # Category:Unidentified plants
#    def _guess_Classify_Plants(self):
#        pass
#    # Category:Unidentified coats of arms
#    def _guess_Classify_CoatsOfArms(self):
#        pass
#    # Category:Unidentified buildings
#    def _guess_Classify_Buildings(self):
#        pass
#    # Category:Unidentified trains
#    def _guess_Classify_Trains(self):
#        pass
#    # Category:Unidentified automobiles
#    def _guess_Classify_Automobiles(self):
#        pass
#    # Category:Unidentified buses
#    def _guess_Classify_Buses(self):
#        pass

    # Category:Human legs
    def _guess_legs_HumanLegs(self):
        result = self._info_filter['Legs']
 
        return (u'Human legs', ((len(result) == 1) and (result[0]['Coverage'] >= .40)))

    # Category:Human torsos
    def _guess_torsos_HumanTorsos(self):
        result = self._info_filter['Torsos']
 
        return (u'Human torsos', ((len(result) == 1) and (result[0]['Coverage'] >= .40)))

    # Category:Automobiles
    def _guess_automobiles_Automobiles(self):
        result = self._info_filter['Automobiles']
 
        return (u'Automobiles', ((len(result) == 1) and (result[0]['Coverage'] >= .40)))

    ## Category:Hands
    #def _guess_hands_Hands(self):
    #    result = self._info_filter['Hands']
    #
    #    return (u'Hands', ((len(result) == 1) and (result[0]['Coverage'] >= .50)))

    # Category:Black     (  0,   0,   0)
    # Category:Blue‎      (  0,   0, 255)
    # Category:Brown     (165,  42,  42)
    # Category:Green     (  0, 255,   0)
    # Category:Orange    (255, 165,   0)
    # Category:Pink‎      (255, 192, 203)
    # Category:Purple    (160,  32, 240)
    # Category:Red‎       (255,   0,   0)
    # Category:Turquoise ( 64, 224, 208)
    # Category:White‎     (255, 255, 255)
    # Category:Yellow    (255, 255,   0)
    # http://www.farb-tabelle.de/en/table-of-color.htm
    #def _collectColor(self):
    #def _cat_color_Black(self):
    #    info = self._info_filter['ColorRegions']
    #    for item in info:
    #        if (u'Black' == item[u'Color']):
    #            return (u'Black', True)
    #    return (u'Black', False)

    def __cat_color_general(self, col):
        info = self._info_filter['ColorRegions']
        for item in info:
            if (col == item[u'Color']):
                return (col, True)
        return (col, False)

    _cat_color_Black     = lambda self: self.__cat_color_general(u'Black')
    _cat_color_Blue      = lambda self: self.__cat_color_general(u'Blue')
    _cat_color_Brown     = lambda self: self.__cat_color_general(u'Brown')
    _cat_color_Green     = lambda self: self.__cat_color_general(u'Green')
    _cat_color_Orange    = lambda self: self.__cat_color_general(u'Orange')
    _cat_color_Pink      = lambda self: self.__cat_color_general(u'Pink')
    _cat_color_Purple    = lambda self: self.__cat_color_general(u'Purple')
    _cat_color_Red       = lambda self: self.__cat_color_general(u'Red')
    _cat_color_Turquoise = lambda self: self.__cat_color_general(u'Turquoise')
    _cat_color_White     = lambda self: self.__cat_color_general(u'White')
    _cat_color_Yellow    = lambda self: self.__cat_color_general(u'Yellow')


# all classification and categorization methods and definitions - SVM variation
#  use 'pyml' SVM (libsvm) classifier
#  may be 'scikit-learn' or 'opencv' (svm, a.o.) could be of some use too
class CatImages_SVM(CatImages_Default):
    trained_cat = [u'Human_ears', u'Male faces']

    # dummy: deactivated
    def classifyFeatures(self):
        for key in self._info:
            for i in range(len(self._info[key])):
                self._info[key][i]['Confidence'] = 1.0
    
    # (all trained categories)
    # http://scipy-lectures.github.com/advanced/scikit-learn/index.html
    # http://mlpy.sourceforge.net/docs/3.5/index.html
    # http://docs.opencv.org/modules/ml/doc/ml.html
    def _cat_multi_generic(self):
        # IT LOOKS LIKE (MAY BE) scikit-learn IS BETTER AND HAS MORE OPTIONS THAN pyml ... ?!!!

        # create classifier feature set
        # !!!currently number of detected features is used only -> lots of room for improvements!!!
        features = []
        for key in sorted(self._info):
            #print key, len(self._info[key]), self._info[key]
            features.append( len(self._info[key]) )
        features = np.array(features)

        linear_svm = mlpy.LibSvm().load_model('cache/test.csf')
        yp  = linear_svm.pred(features)
        cat = self.trained_cat[int(yp)-1]
        #print linear_svm.labels()
        # confidence of match?
 
        return (cat, True)


# Image by content categorization derived from 'checkimages.py'.
class CatImagesBot(checkimages.checkImagesBot, CatImages_Default):
#class CatImagesBot(checkimages.checkImagesBot, CatImages_SVM):
#    def __init__(self, site, logFulNumber = 25000, sendemailActive = False,
#                 duplicatesReport = False, logFullError = True): pass
#    def setParameters(self, imageName): pass

    # or may be '__init__' ... ???
    def load_licenses(self):
        #pywikibot.output(u'\n\t...Listing the procedures available...\n')
        pywikibot.output(u'\n\t...Listing the procedures used...\n')
        
        self._funcs = {'filter': [], 'cat': [], 'addcat': [], 'guess': []}

        for item in dir(self):
            s = item.split('_')
            if (len(s) < 3) or (s[1] not in self._funcs) or (s[2] in self.ignore):
                continue
            pywikibot.output( item )
            self._funcs[s[1]].append( item )

        self.tmpl_available_spec = tmpl_available_spec
        gen = pagegenerators.PrefixingPageGenerator(prefix = u'Template:FileContentsByBot/')
        buf = []
        for item in gen:
            item = item.title()
            if (item[-4:] == "/doc"):           # all docs
                continue
            item = os.path.split(item)[1]
            if (item[0].lower() == item[0]):    # e.g. 'generic'
                continue
            buf.append( item )
        if buf:
            self.tmpl_available_spec = buf
            pywikibot.output( u'\n\t...Following specialized templates found, check them since they are used now...\n' )
            pywikibot.output( u'tmpl_available_spec = [ %s ]\n' % u", ".join(buf) )

        return []

    def downloadImage(self):
        #print self.image_path
        pywikibot.output(u'Processing media %s ...' % self.image.title(asLink=True))

        self.image_filename  = os.path.split(self.image.fileUrl())[-1]
        self.image_fileext   = os.path.splitext(self.image_filename)[1]
        self.image_path      = urllib2.quote(os.path.join(scriptdir, ('cache/' + self.image_filename[-128:])))
        
        self.image_path_JPEG = self.image_path + u'.jpg'
        
        self._wikidata = self.image._latestInfo # all info wikimedia got from content (mime, sha1, ...)
        #print self._wikidata
        #print self._wikidata['mime']
        #print self._wikidata['sha1']
        #print self._wikidata['metadata']
        #for item in self._wikidata['metadata']:
        #    print item['name'], item['value']
        
        if not os.path.exists(self.image_path):
            pywikibot.get_throttle()
            f_url, data = self.site.getUrl(self.image.fileUrl(), no_hostname=True, 
                                           back_response=True)
            # needed patch for 'getUrl' applied upstream in r10441
            # (allows to re-read from back_response)
            data = f_url.read()
            del f_url   # free some memory (no need to keep a copy...)
    
            f = open(self.image_path, 'wb')
            f.write( data )
            f.close()

        # 'magic' (libmagic)
        m = magic.open(magic.MAGIC_MIME)    # or 'magic.MAGIC_NONE'
        m.load()
        self.image_mime = re.split('[/;\s]', m.file(self.image_path))
        #self.image_size = (None, None)
        mime = mimetypes.guess_all_extensions('%s/%s' % tuple(self.image_mime[0:2]))
        if self.image_fileext.lower() not in mime:
            pywikibot.output(u'WARNING: File extension does not match MIME type! File extension should be %s.' % mime)

        # SVG: rasterize the SVG to bitmap (MAY BE GET FROM WIKI BY DOWNLOAD?...)
        # (Mediawiki uses librsvg too: http://commons.wikimedia.org/wiki/SVG#SVGs_in_MediaWiki)
        # http://stackoverflow.com/questions/6589358/convert-svg-to-png-in-python
        # http://cairographics.org/pythoncairopil/
        # http://cairographics.org/pyrsvg/
        # http://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
        self.image_size = (None, None)
        if   self.image_fileext == u'.svg':
            try:
                svg = rsvg.Handle(self.image_path)
                img = cairo.ImageSurface(cairo.FORMAT_ARGB32, svg.props.width, svg.props.height)
                ctx = cairo.Context(img)
                svg.render_cairo(ctx)
                #img.write_to_png("svg.png")
                #Image.frombuffer("RGBA",( img.get_width(),img.get_height() ),
                #             img.get_data(),"raw","RGBA",0,1).save(self.image_path_JPEG, "JPEG")
                png = Image.frombuffer("RGBA",( img.get_width(),img.get_height() ),
                                   img.get_data(),"raw","RGBA",0,1)
                background = Image.new("RGB", png.size, (255, 255, 255))
                background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
                background.save(self.image_path_JPEG, "JPEG")

                self.image_size = (svg.props.width, svg.props.height)
            except MemoryError:
                self.image_path_JPEG = self.image_path
            except SystemError:
                self.image_path_JPEG = self.image_path
        elif self.image_fileext == u'.xcf':
            # Very few programs other than GIMP read XCF files. This is by design
            # from the GIMP developers, the format is not really documented or
            # supported as a general-purpose file format.
            # Commons uses ImageMagick, thus we have EXACTLY THE SAME support!
            # (can also be a drawback, e.g. when the library is buggy...)
            proc = Popen("convert %s %s" % (self.image_path, self.image_path_JPEG),
                         shell=True, stderr=PIPE)#.stderr.read()
            proc.wait()
            if   proc.returncode == 127:
                raise ImportError("convert (ImageMagick) not found!")
            elif proc.returncode:
                self.image_path_JPEG = self.image_path

            #data = Popen("identify -verbose info: %s" % self.image_path,
            #             shell=True, stderr=PIPE).stderr.read()
            #print data
            self.image_size = Image.open(self.image_path_JPEG).size
        else:
            try:
                im = Image.open(self.image_path) # might be png, gif etc, for instance
                #im.thumbnail(size, Image.ANTIALIAS) # size is 640x480
                im.convert('RGB').save(self.image_path_JPEG, "JPEG")

                self.image_size = im.size
            except IOError, e:
                if 'image file is truncated' in str(e):
                    # im object has changed due to exception raised
                    im.convert('RGB').save(self.image_path_JPEG, "JPEG")

                    self.image_size = im.size
                else:
                    try:
                        # since opencv might still work, try this as fall-back
                        img = cv2.imread( self.image_path, 1 )
                        cv2.imwrite(self.image_path_JPEG, img)

                        self.image_size = (img.shape[1], img.shape[0])
                    except:
                        self.image_path_JPEG = self.image_path
            except:
                self.image_path_JPEG = self.image_path

        # PDF: support extract text and images
        # (Mediawiki uses ghostscript: https://www.mediawiki.org/wiki/Extension:PdfHandler#Pre-requisites)
        # http://vermeulen.ca/python-pdf.html
        # http://code.activestate.com/recipes/511465-pure-python-pdf-to-text-converter/
        # http://stackoverflow.com/questions/25665/python-module-for-converting-pdf-to-text
        if self.image_fileext == u'.pdf':
            pass
        
        # FULL TIFF support (e.g. group4)
        # http://code.google.com/p/pylibtiff/

    # LOOK ALSO AT: checkimages.CatImagesBot.checkStep
    # (and category scripts/bots too...)
    def checkStep(self):
        self.thrshld = self._thrshld_default

        self._info         = {}     # used for LOG/DEBUG OUTPUT ONLY
        self._info_filter  = {}     # used for CATEGORIZATION
        self._result_check = []
        self._result_add   = []
        self._result_guess = []

        # flush internal buffers
        for attr in ['_buffer_EXIF', '_buffer_FFMPEG', '_buffer_Geometry']:#, '_content_text']:
            if hasattr(self, attr):
                delattr(self, attr)

        # gather all features (information) related to current image
        self.gatherFeatures()

        # classification of detected features (should use RTrees, KNearest, Boost, SVM, MLP, NBayes, ...)
        # ??? (may be do this in '_cat_...()' or '_filter_...()' ?!?...)
        # http://opencv.itseez.com/doc/tutorials/ml/introduction_to_svm/introduction_to_svm.html
        # http://stackoverflow.com/questions/8687885/python-opencv-svm-implementation
        # https://code.ros.org/trac/opencv/browser/trunk/opencv/samples/python2/letter_recog.py?rev=6480
        self.classifyFeatures()      # assign confidences
        # replace/improve this with RTrees, KNearest, Boost, SVM, MLP, NBayes, ...

        # information template: use filter to select from gathered features
        #                       the ones that get reported
        self._info_filter = {}
        for item in self._funcs['filter']:
            self._info_filter.update( getattr(self, item)() )

        # categorization: use explicit searches for classification (rel = ?)
        for item in self._funcs['cat']:
            (cat, rel) = getattr(self, item)()
            #print cat, result, len(result)
            if rel:
                self._result_check.append( cat )
        self._result_check = list(set(self._result_check))

        # categorization: conditional (only if the ones before are present)
        # (does not trigger report to page)
        for item in self._funcs['addcat']:
            (cat, rel) = getattr(self, item)()
            #print cat, result, len(result)
            if rel:
                self._result_add.append( cat )
        self._result_add = list(set(self._result_add))

        # categorization: use guesses for unreliable classification (rel = 0.1)
        if not useGuesses:
            return self._result_check
        for item in self._funcs['guess']:
            (cat, rel) = getattr(self, item)()
            #print cat, result, len(result)
            if rel:
                self._result_guess.append( cat )

        return self._result_check

    def tag_image(self):
        self.clean_cache()

        #if not self._existInformation(self._info_filter):  # information available?
        if not (self._result_check + self._result_guess):   # category available?
            return False

        pywikibot.get_throttle()
        content = self.image.get()

        content = self._append_to_template(content, u"Information", tmpl_FileContentsByBot)
        for i, key in enumerate(self._info_filter):
            item = self._info_filter[key]

            info = self._make_infoblock(key, item)
            if info:
                content = self._append_to_template(content, u"FileContentsByBot", info)

        tags = set([])
        for i, cat in enumerate(list(set(self._result_check + self._result_add))):
            tags.add( u"[[:Category:%s]]" % cat )
            content = pywikibot.replaceCategoryLinks(content, [cat], site=self.site, addOnly=True)

        content = pywikibot.replaceCategoryLinks( content, 
                list(set(pywikibot.getCategoryLinks(content, site=self.site))),
                site=self.site )
        content = self._remove_category_or_template(content, u"Uncategorized")  # template
        content = self._add_template(content, u"Check categories|year={{subst:#time:Y}}|month={{subst:#time:F}}|day={{subst:#time:j}}|category=[[Category:Categorized by DrTrigonBot]]", top=True)

        for i, cat in enumerate(self._result_guess):
            content += u"\n<!--DrTrigonBot-guess-- [[Category:%s]] -->" % cat

        pywikibot.output(u"--- " * 20)
        pywikibot.output(content)
        pywikibot.output(u"--- " * 20)
        pywikibot.put_throttle()
        self.image.put( content, comment="bot automatic categorization; adding %s" % u", ".join(tags),
                                 botflag=False )

# TODO: (work-a-round if https://bugzilla.wikimedia.org/show_bug.cgi?id=6421 not solved)
#        if hasattr(self, '_content_text'):
#            textpage = pywikibot.Page(self.site, os.path.join(self.image.title(), u'Contents/Text'))
#            textpage.put( self._content_text, comment="bot adding content from %s" % textpage.title(asLink=True),
#                                              botflag=False )

        return True

    def log_output(self):
        # ColorRegions always applies here since there is at least 1 (THE average) color...
        ignore = ['Properties', 'ColorAverage', 'ColorRegions', 'Geometry']
        #if not self._existInformation(self._info):  # information available?
        # information available? AND/OR category available?
        if not (self._existInformation(self._info, ignore = ignore) or self._result_check):
            return u""

        ret  = []
        ret.append( u"" )
        ret.append( u"== [[:%s]] ==" % self.image.title() )
        ret.append( u'{|' )
        ret.append( u'|<div style="position:relative;">' )
        ret.append( u"[[%s|200px]]" % self.image.title() )
        ret.append( self._make_markerblock(self._info[u'Faces'], 200.,
                                           structure=['Position', 'Eyes', 'Mouth', 'Nose']) )
        ret.append( self._make_markerblock(self._info[u'People'], 200.,
                                           line='dashed') )
        ret.append( u"</div>" )
        ret.append( u'|<div style="position:relative;">' )
        ret.append( u"[[%s|200px]]" % self.image.title() )
        ret.append( self._make_markerblock(self._info[u'ColorRegions'], 200.) )
        ret.append( self._make_markerblock(self._info[u'OpticalCodes'], 200.,
                                           line='dashed') )
        ret.append( u"</div>" )
        ret.append( u'|<div style="position:relative;">' )
        ret.append( u"[[%s|200px]]" % self.image.title() )
        ret.append( self._make_markerblock(self._info[u'Ears'], 200.) )
        ret.append( self._make_markerblock(self._info[u'Eyes'], 200.) )
        ret.append( self._make_markerblock(self._info[u'Legs'], 200.,
                                           line='dashed') )
        ret.append( self._make_markerblock(self._info[u'Torsos'], 200.,
                                           line='dashed') )
        ret.append( self._make_markerblock(self._info[u'Automobiles'], 200.,
                                           line='dashed') )
        #ret.append( self._make_markerblock(self._info[u'Hands'], 200.,
        #                                   line='dashed') )
        ret.append( u"</div>" )
        ret.append( u'|}' )

        color = {True: "rgb(0,255,0)", False: "rgb(255,0,0)"}[bool(self._result_check + self._result_guess)]
        ret.append( u"<div style='background:%s'>'''automatic categorization''': %s</div>" % (color, u", ".join(list(set(self._result_check + self._result_add)))) )

        buf = []
        for i, key in enumerate(self._info):
            item = self._info[key]

            info = self._make_infoblock(key, item, [])
            if info:
                buf.append( info )
        ret.append( tmpl_FileContentsByBot[3:] + u"\n" + u"\n".join( buf ) + u"\n}}" )

        return u"\n".join( ret )

    def clean_cache(self):
        if os.path.exists(self.image_path):
            os.remove( self.image_path )
        if os.path.exists(self.image_path_JPEG):
            os.remove( self.image_path_JPEG )
        #image_path_new = self.image_path_JPEG.replace(u"cache/", u"cache/0_DETECTED_")
        #if os.path.exists(image_path_new):
        #    os.remove( image_path_new )

    # LOOK ALSO AT: checkimages.CatImagesBot.report
    def report(self):
        tagged = self.tag_image()
        logged = self.log_output()
        return (tagged, logged)

    def _make_infoblock(self, cat, res, tmpl_available=None):
        if not res:
            return u''

        if (tmpl_available == None):
            tmpl_available = self.tmpl_available_spec

        generic = (cat not in tmpl_available)
        titles = res[0].keys()
        if not titles:
            return u''

        result = []
        #result.append( u'{{(!}}style="background:%s;"' % {True: 'green', False: 'red'}[report] )
        if generic:
            result.append( u"{{FileContentsByBot/generic|name=%s|" % cat )
            buf = dict([ (key, []) for key in titles ])
            for item in res:
                for key in titles:
                    buf[key].append( self._output_format(item[key]) )
            for key in titles:
                result.append( u"  {{FileContentsByBot/generic|name=%s|value=%s}}" % (key, u"; ".join(buf[key])) )
        else:
            result.append( u"{{FileContentsByBot/%s|" % cat )
            for item in res:
                result.append( u"  {{FileContentsByBot/%s" % cat )
                for key in titles:
                    if item[key]:   # (work-a-round for empty 'Eyes')
                        result.append( self._output_format_flatten(key, item[key]) )
                result.append( u"  }}" )
        result.append( u"}}" )

        return u"\n".join( result )

    def _output_format(self, value):
        if (type(value) == type(float())):
            # round/strip floats
            return "%.3f" % value
        else:
            # output string representation of variable
            return str(value)

    def _output_format_flatten(self, key, value):
        # flatten structured varible recursively
        if (type(value) == type(tuple())) or (type(value) == type(list())):
            buf = []
            for i, t in enumerate(value):
                buf.append( self._output_format_flatten(key + (u"-%02i" % i), t) )
            return u"\n".join( buf )
        else:
            # end of recursion
            return u"  | %s = %s" % (key, self._output_format(value))

    def _make_markerblock(self, res, size, structure=['Position'], line='solid'):
        # same as in '_detect_Faces_CV'
        colors = [ (0,0,255),
            (0,128,255),
            (0,255,255),
            (0,255,0),
            (255,128,0),
            (255,255,0),
            (255,0,0),
            (255,0,255) ]
        result = []
        for i, r in enumerate(res):
            if ('RGB' in r):
                color = list(np.array((255,255,255))-np.array(r['RGBref']))
            else:
                color = list(colors[i%8])
            color.reverse()
            color = u"%02x%02x%02x" % tuple(color)
            
            #scale = r['size'][0]/size
            scale = self.image_size[0]/size
            f     = list(np.array(r[structure[0]])/scale)
            
            result.append( u'<div class="%s-marker" style="position:absolute; left:%ipx; top:%ipx; width:%ipx; height:%ipx; border:2px %s #%s;"></div>' % tuple([structure[0].lower()] + f + [line, color]) )

            for ei in range(len(structure)-1):
                data = r[structure[ei+1]]
                if data and (not hasattr(data[0], '__iter__')):    # Mouth and Nose are not lists
                    data = [ r[structure[ei+1]] ]
                for e in data:
                    e = list(np.array(e)/scale)
    
                    result.append( u'<div class="%s-marker" style="position:absolute; left:%ipx; top:%ipx; width:%ipx; height:%ipx; border:2px solid #%s;"></div>' % tuple([structure[ei+1].lower()] + e + [color]) )

        return u"\n".join( result )

    # place into 'textlib' (or else e.g. 'catlib'/'templib'...)
    def _remove_category_or_template(self, text, name):
        text = re.sub(u"[\{\[]{2}%s.*?[\}\]]{2}\n?" % name, u"", text)
        return text

    # place into 'textlib'
    def _add_template(self, text, name, params={}, top=False, raw=False):
        if top:
            buf = [(u"{{%s}}" % name), text]
        else:
            if raw:
                buf = [text, name]
            else:
                buf = [text, (u"{{%s}}" % name)]
        return u"\n".join( buf )

    # place into 'textlib' (or else e.g. 'catlib'/'templib'...)
    def _append_to_template(self, text, name, append):
        pattern  = re.compile(u"(\{\{%s.*?\n)(\s*\}\}\n{2})" % name, flags=re.S)
        template = pattern.search(text).groups()
        
        template = u"".join( [template[0], append, u"\n", template[1]] )
        
        text = pattern.sub(template, text)
        return text

    # gather data from all information interfaces
    def gatherFeatures(self):
        # Image size
        self._detect_Properties_PIL()
        
        self._info['Faces'] = []
        # Faces (extract EXIF data)
        self._detect_Faces_EXIF()
        # Faces and eyes (opencv pre-trained haar)
        self._detect_Faces_CV()
        # exclude duplicates (CV and EXIF)
        faces = [item['Position'] for item in self._info['Faces']]
        for i in self._util_merge_Regions(faces)[1]:
            del self._info['Faces'][i]

        # Segments and colors
        self._detect_SegmentColors_JSEGnPIL()
        # Average color
        self._detect_AverageColor_PILnCV()

        # People/Pedestrian (opencv pre-trained hog and haarcascade)
        self._detect_People_CV()

        # Geometric object (opencv hough line, circle, edges, corner, ...)
        self._detect_Geometry_CV()

        # general (opencv pre-trained, third-party and self-trained haar
        # and cascade) classification
        # http://www.computer-vision-software.com/blog/2009/11/faq-opencv-haartraining/
        for cf in self.cascade_files:
            self._detect_Trained_CV(*cf)

        # optical and other text recognition (tesseract & ocropus, ...)
        self._detect_EmbeddedText_poppler()
#        self._recognize_OpticalText_ocropus()
        # (may be just classify as 'contains text', may be store text, e.g. to wikisource)

        # barcode and Data Matrix recognition (libdmtx/pydmtx, zbar, gocr?)
        self._recognize_OpticalCodes_dmtxNzbar()

        # Chessboard (opencv reference detector)
        self._detect_Chessboard_CV()

        # general (self-trained) detection WITH classification (BoW)
        # uses feature detection (SIFT, SURF, ...) AND classification (SVM, ...)
#        self._detectclassify_ObjectAll_CV()

        # general handling of all audio and video formats
        self._detect_Streams_FFMPEG()

        # general file EXIF history information
        self._detect_History_EXIF()

        # general audio feature extraction
#        self._detect_AudioFeatures_YAAFE()

    def _existInformation(self, info, ignore = ['Properties', 'ColorAverage']):
        result = []
        for item in info:
            if item in ignore:
                continue
            if info[item]:
                result.append( item )
        return result

    def _filter_Properties(self):
        # >>> never drop <<<
        result = self._info['Properties']
        return {'Properties': result}

    def _filter_Faces(self):
        result = self._info['Faces']
        if (len(result) < self._thrhld_group_size):
            buf = []
            for item in self._info['Faces']:
                # >>> drop if below thrshld <<<
                if (item['Confidence'] >= self.thrshld):
                    buf.append( item )
            result = buf
        return {'Faces': result}

    def _filter_People(self):
        result = self._info['People']
        if (len(result) < self._thrhld_group_size):
            buf = []
            for item in self._info['People']:
                # >>> drop if below thrshld <<<
                if (item['Confidence'] >= self.thrshld):
                    buf.append( item )
            result = buf
        return {'People': result}

    def _filter_ColorRegions(self):
        #result = {}
        result = []
        for item in self._info['ColorRegions']:
            ## >>> drop wrost ones... (ignore all below 0.2) <<<
            #if (result.get(item['Color'], {'Confidence': 0.2})['Confidence'] < item['Confidence']):
            #    result[item['Color']] = item
            # >>> drop if below thrshld <<<
            if (item['Confidence'] >= self.thrshld):
                result.append( item )
        #return {'ColorRegions': [result[item] for item in result]}
        return {'ColorRegions': result}

    def _filter_ColorAverage(self):
        # >>> never drop <<<
        result = self._info['ColorAverage']
        return {'ColorAverage': result}

    def _filter_OpticalCodes(self):
        # use all, since detection should be very reliable
        #result = self._info['OpticalCodes']
        result = []
        for item in self._info['OpticalCodes']:
            # >>> drop if below thrshld <<<
            if (item['Confidence'] >= self.thrshld):
                result.append( item )
        return {'OpticalCodes': result}

    def _filter_Chessboard(self):
        # use all, since detection should be very reliable
        result = self._info['Chessboard']
        return {'Chessboard': result}

    def _filter_Text(self):
        # use all, since detection should be very reliable
        result = self._info['Text']
        return {'Text': result}

    def _filter_Legs(self):
        result = []
        for item in self._info['Legs']:
            # >>> drop if below thrshld <<<
            if (item['Confidence'] >= self.thrshld):
                result.append( item )
        return {'Legs': result}

    def _filter_Torsos(self):
        result = []
        for item in self._info['Torsos']:
            # >>> drop if below thrshld <<<
            if (item['Confidence'] >= self.thrshld):
                result.append( item )
        return {'Torsos': result}

    def _filter_Ears(self):
        result = []
        for item in self._info['Ears']:
            # >>> drop if below thrshld <<<
            if (item['Confidence'] >= self.thrshld):
                result.append( item )
        return {'Ears': result}

    def _filter_Eyes(self):
        result = []
        for item in self._info['Eyes']:
            # >>> drop if below thrshld <<<
            if (item['Confidence'] >= self.thrshld):
                result.append( item )
        return {'Eyes': result}

    def _filter_Automobiles(self):
        result = []
        for item in self._info['Automobiles']:
            # >>> drop if below thrshld <<<
            if (item['Confidence'] >= self.thrshld):
                result.append( item )
        return {'Automobiles': result}

    def _filter_Streams(self):
        # use all, (should be reliable)
        result = self._info['Streams']
        return {'Streams': result}

    #def _filter_Geometry(self):
    #    result = []
    #    for item in self._info['Geometry']:
    #        # >>> drop if below thrshld <<<
    #        if (item['Confidence'] >= self.thrshld):
    #            result.append( item )
    #    return {'Geometry': result}

    #def _filter_Hands(self):
    #    result = []
    #    for item in self._info['Hands']:
    #        # >>> drop if below thrshld <<<
    #        if (item['Confidence'] >= self.thrshld):
    #            result.append( item )
    #    return {'Hands': result}

#    def _filter_Classify(self):
#        from operator import itemgetter
#        result = sorted(self._info['Classify'][0].items(), key=itemgetter(1))
#        result.reverse()
#        pywikibot.output(u' Best: %s' % result[:3] )
#        pywikibot.output(u'Worst: %s' % result[-3:] )
#
#        # >>> dummy: drop all (not reliable yet since untrained) <<<
#        return {'Classify': []}


def main():
    """ Main function """
    global useGuesses
    # Command line configurable parameters
    limit = 150 # How many images to check?
#    untagged = False # Use the untagged generator
    sendemailActive = False # Use the send-email
    train = False
    generator = None

    # default
    if len(sys.argv) < 2:
        sys.argv += ['-cat']

    # debug:    'python catimages.py -debug'
    # run/test: 'python catimages.py [-start:File:abc]'
    sys.argv += ['-family:commons', '-lang:commons']
    #sys.argv += ['-noguesses']

    # try to resume last run and continue
    if os.path.exists( os.path.join(scriptdir, 'cache/catimages_start') ):
        shutil.copy2(os.path.join(scriptdir, 'cache/catimages_start'), os.path.join(scriptdir, 'cache/catimages_start.bak'))
        posfile = open(os.path.join(scriptdir, 'cache/catimages_start'), "r")
        firstPageTitle = posfile.read().decode('utf-8')
        posfile.close()
    else:
        firstPageTitle = None

    # Here below there are the parameters.
    for arg in pywikibot.handleArgs():
        if arg.startswith('-limit'):
            if len(arg) == 7:
                limit = int(pywikibot.input(u'How many files do you want to check?'))
            else:
                limit = int(arg[7:])
#        elif arg == '-sendemail':
#            sendemailActive = True
        elif arg.startswith('-start'):
            if len(arg) == 6:
                firstPageTitle = None
            elif len(arg) > 6:
                firstPageTitle = arg[7:]
            #firstPageTitle = firstPageTitle.split(":")[1:]
            #generator = pywikibot.getSite().allpages(start=firstPageTitle, namespace=6)
        elif arg.startswith('-cat'):
            if len(arg) == 4:
                catName = u'Media_needing_categories'
            elif len(arg) > 4:
                catName = str(arg[5:])
            catSelected = catlib.Category(pywikibot.getSite(), 'Category:%s' % catName)
            generator = pagegenerators.CategorizedPageGenerator(catSelected, recurse = True)
#        elif arg.startswith('-untagged'):
#            untagged = True
#            if len(arg) == 9:
#                projectUntagged = str(pywikibot.input(u'In which project should I work?'))
#            elif len(arg) > 9:
#                projectUntagged = str(arg[10:])
        elif arg == '-noguesses':
            useGuesses = False
        elif arg.startswith('-single'):
            if len(arg) > 7:
                pageName = unicode(arg[8:])
            if 'File:' not in pageName:
                pageName = 'File:%s' % pageName
            generator = [ pywikibot.Page(pywikibot.getSite(), pageName) ]
            firstPageTitle = None
        elif arg.startswith('-train'):
            train = True
            generator = None

    # Understand if the generator is present or not.
    if not generator:
        pywikibot.output(u'no generator defined... EXIT.')
        sys.exit()
            
    # Define the site.
    site = pywikibot.getSite()

    # Block of text to translate the parameters set above.
    image_old_namespace = u"%s:" % site.image_namespace()
    image_namespace = u"File:"

    # A little block-statement to ensure that the bot will not start with en-parameters
    if site.lang not in project_inserted:
        pywikibot.output(u"Your project is not supported by this script. You have to edit the script and add it!")
        return

    # Defing the Main Class.
    Bot = CatImagesBot(site, sendemailActive = sendemailActive,
                       duplicatesReport = False, logFullError = False)
#    # Untagged is True? Let's take that generator
#    if untagged == True:
#        generator =  Bot.untaggedGenerator(projectUntagged, limit)
    # Ok, We (should) have a generator, so let's go on.
    # Take the additional settings for the Project
    Bot.takesettings()

    # do classifier training on good (homgenous) commons categories
    if train:
        trainbot(generator, Bot, image_old_namespace, image_namespace)
        return

    # Not the main, but the most important loop.
    outresult = []
    for image in generator:
        if firstPageTitle:
            if (image.title() == firstPageTitle):
                pywikibot.output( u"found last page '%s' ..." % image.title() )
                firstPageTitle = None
                continue
            else:
                #pywikibot.output( u"skipping page '%s' ..." % image.title() )
                continue

        # recover from hard crash in the run before, thus skip one more page
        if os.path.exists( os.path.join(scriptdir, 'cache/catimages_recovery') ):
            pywikibot.output( u"trying to recover from hard crash, skipping page '%s' ..." % image.title() )
            disable_recovery()

            # in case the next one has a hard-crash too...
            posfile = open(os.path.join(scriptdir, 'cache/catimages_start'), "w")
            posfile.write( image.title().encode('utf-8') )
            posfile.close()

            continue

        #comment = None # useless, also this, let it here for further developments
        try:
            imageName = image.title().split(image_namespace)[1] # Deleting the namespace (useless here)
        except IndexError:# Namespace image not found, that's not an image! Let's skip...
            try:
                imageName = image.title().split(image_old_namespace)[1]
            except IndexError:
                pywikibot.output(u"%s is not a file, skipping..." % image.title())
                continue
        Bot.setParameters(imageName) # Setting the image for the main class
        try:
            Bot.downloadImage()
        except IOError, err:
            # skip if download not possible
            pywikibot.output(u"WARNING: %s, skipped..." % err)
            continue
        except Exception, err:
            # skip on any unexpected error, but report it
            pywikibot.output(u"ERROR: %s" % err)
            pywikibot.output(u"ERROR: was not able to process page %s !!!\n" %\
                             image.title(asLink=True))
            continue
        resultCheck = Bot.checkStep()
        tagged = False
        try:
            (tagged, ret) = Bot.report()
            if ret:
                outresult.append( ret )
        except AttributeError:
            pywikibot.output(u"ERROR: was not able to process page %s !!!\n" %\
                             image.title(asLink=True))
        limit += -1
        if not tagged:
            posfile = open(os.path.join(scriptdir, 'cache/catimages_start'), "w")
            posfile.write( image.title().encode('utf-8') )
            posfile.close()
        if limit <= 0:
            break
        if resultCheck:
            continue

    if outresult:
        outpage = pywikibot.Page(site, u"User:DrTrigon/User:DrTrigonBot/logging")
        #outresult = [ outpage.get() ] + outresult   # append to page
        outpage.put( u"\n".join(outresult), comment="bot writing log for last run" )
        if pywikibot.simulate:
            #print u"--- " * 20
            #print u"--- " * 20
            #print u"\n".join(outresult)
            posfile = open(os.path.join(scriptdir, 'cache/catimages.log'), "a")
            posfile.write( u"\n".join(outresult) )
            posfile.close()

# http://scipy-lectures.github.com/advanced/scikit-learn/index.html
# http://mlpy.sourceforge.net/docs/3.5/index.html
# http://docs.opencv.org/modules/ml/doc/ml.html
# train pyml (svm), opencv BoW and haarcascade classifiers
# choose a good and meaningful featureset from extracted (better than actual one)
def trainbot(generator, Bot, image_old_namespace, image_namespace):
    # IT LOOKS LIKE (MAY BE) scikit-learn IS BETTER AND HAS MORE OPTIONS THAN pyml ... ?!!!

    # gather training dataset from wiki commons categories
    trainset = []
    for i, catName in enumerate(Bot.trained_cat):
        catSelected = catlib.Category(pywikibot.getSite(), 'Category:%s' % catName)
        generator = pagegenerators.CategorizedPageGenerator(catSelected)

        for image in generator:
            try:
                imageName = image.title().split(image_namespace)[1] # Deleting the namespace (useless here)
            except IndexError:# Namespace image not found, that's not an image! Let's skip...
                try:
                    imageName = image.title().split(image_old_namespace)[1]
                except IndexError:
                    pywikibot.output(u"%s is not a file, skipping..." % image.title())
                    continue
            Bot.setParameters(imageName) # Setting the image for the main class
            try:
                Bot.downloadImage()
            except IOError, err:
                # skip if download not possible
                pywikibot.output(u"WARNING: %s, skipped..." % err)
                continue
            except Exception, err:
                # skip on any unexpected error, but report it
                pywikibot.output(u"ERROR: %s" % err)
                pywikibot.output(u"ERROR: was not able to process page %s !!!\n" %\
                                 image.title(asLink=True))
                continue

            # gather all features (information) related to current image
            Bot._info = {}
            Bot.gatherFeatures()
    
            # create classifier feature set
            # !!!currently number of detected features is used only -> lots of room for improvements!!!
            # choose a good and meaningful featureset from extracted (better than actual one)
            features = []
            for key in sorted(Bot._info):
                #print key, len(self._info[key]), self._info[key]
                features.append( len(Bot._info[key]) )
            features.append( i+1 )      # category id (returned by predictor later)
            #print features
            trainset.append( features )

    trainset = np.array(trainset)
    cols = trainset.shape[1]

    # http://mlpy.sourceforge.net/docs/3.5/tutorial.html
    import matplotlib.pyplot as plt # required for plotting

    ##iris = np.loadtxt('iris.csv', delimiter=',')
    ##x, y = iris[:, :4], iris[:, 4].astype(np.int) # x: (observations x attributes) matrix, y: classes (1: setosa, 2: versicolor, 3: virginica)
    #trainset = np.loadtxt('cache/test.csv', delimiter=' ')
    #cols = trainset.shape[1]
    #print trainset
    x, y = trainset[:, :(cols-1)], trainset[:, (cols-1)].astype(np.int) # x: (observations x attributes) matrix, y: classes (1: setosa, 2: versicolor, 3: virginica)
    pywikibot.output(x.shape)
    pywikibot.output(y.shape)
    
    # Dimensionality reduction by Principal Component Analysis (PCA)
    pca = mlpy.PCA() # new PCA instance
    pca.learn(x) # learn from data
    z = pca.transform(x, k=2) # embed x into the k=2 dimensional subspace
    pywikibot.output(z.shape)
    
    plt.set_cmap(plt.cm.Paired)
    fig1 = plt.figure(1)
    title = plt.title("PCA on dataset")
    plot = plt.scatter(z[:, 0], z[:, 1], c=y)
    labx = plt.xlabel("First component")
    laby = plt.ylabel("Second component")
    plt.show()
    
    # Learning by Kernel Support Vector Machines (SVMs) on principal components
    linear_svm = mlpy.LibSvm(kernel_type='linear') # new linear SVM instance
    linear_svm.learn(z, y) # learn from principal components
    
    # !!! train also BoW (bag-of-words) in '_detectclassify_ObjectAll_CV' resp. 'opencv.BoWclassify.main' !!!
    
    xmin, xmax = z[:,0].min()-0.1, z[:,0].max()+0.1
    ymin, ymax = z[:,1].min()-0.1, z[:,1].max()+0.1
    xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.01), np.arange(ymin, ymax, 0.01))
    zgrid = np.c_[xx.ravel(), yy.ravel()]

    yp = linear_svm.pred(zgrid)
    
    plt.set_cmap(plt.cm.Paired)
    fig2 = plt.figure(2)
    title = plt.title("SVM (linear kernel) on principal components")
    plot1 = plt.pcolormesh(xx, yy, yp.reshape(xx.shape))
    plot2 = plt.scatter(z[:, 0], z[:, 1], c=y)
    labx = plt.xlabel("First component")
    laby = plt.ylabel("Second component")
    limx = plt.xlim(xmin, xmax)
    limy = plt.ylim(ymin, ymax)
    plt.show()
    
    linear_svm.save_model('cache/test.csf')
    pywikibot.output(u'Linear SVM model stored to %s.' % 'cache/test.csf')


# for functions in C/C++ that might crash hard without any exception throwed
# e.g. an abort due to an assert or something else
def enable_recovery():
    recoveryfile = open(os.path.join(scriptdir, 'cache/catimages_recovery'), "w")
    recoveryfile.write('')
    recoveryfile.close()

def disable_recovery():
    if os.path.exists( os.path.join(scriptdir, 'cache/catimages_recovery') ):
        os.remove( os.path.join(scriptdir, 'cache/catimages_recovery') )


# Main loop will take all the (name of the) images and then i'll check them.
if __name__ == "__main__":
    old = datetime.datetime.strptime(str(datetime.datetime.utcnow()).split('.')[0], "%Y-%m-%d %H:%M:%S") #timezones are UTC
    if sys.exc_info()[0]:   # re-raise ImportError
        raise               #
    try:
        main()
    finally:
        final = datetime.datetime.strptime(str(datetime.datetime.utcnow()).split('.')[0], "%Y-%m-%d %H:%M:%S") #timezones are UTC
        delta = final - old
        secs_of_diff = delta.seconds
        pywikibot.output("Execution time: %s" % secs_of_diff)
        pywikibot.stopme()
