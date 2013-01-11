# -*- coding: utf-8  -*-
""" Module to determine the pywikipedia version (tag, revision and date) """
#
# (C) Merlijn 'valhallasw' van Deen, 2007-2008
# (C) xqt, 2010-2011
# (C) Pywikipedia bot team, 2007-2011
#
# Distributed under the terms of the MIT license.
#
__version__ = '$Id$'

import os
import time
import sys
import urllib
import config

cache = None

class ParseError(Exception):
    """ Parsing went wrong """

def getversion():
    data = getversiondict()
    try:
        rev2 = int(getversion_onlinerepo())
        rev1 = int(data['rev'].split()[0])
        data['cmp_ver'] = 'OUTDATED' if cmp_ver(rev1, rev2)=='<' else 'ok'
    except ParseError:
        data['cmp_ver'] = 'n/a'
    return '%(tag)s (r%(rev)s, %(date)s, %(cmp_ver)s)' % data

def getversiondict():
    global cache
    if cache:
      return cache
    try:
        (tag, rev, date) = getversion_svn()
    except Exception, e:
        try:
            (tag, rev, date) = getversion_nightly()
        except Exception, e:
            import wikipedia
            version = wikipedia.__version__
            if len(version) == 4:
                # the value is most likely '$Id' + '$', it means that
                # wikipedia.py got imported without using svn at all
                cache = dict(tag='', rev='-1 (unknown)', date='0 (unknown)')
                return cache

            id, file, rev, date, ts, author, dollar = version.split(' ')
            tag = ''
            date = time.strptime('%sT%s' % (date, ts), '%Y-%m-%dT%H:%M:%SZ')
            rev += ' (wikipedia.py)'
    datestring = time.strftime('%Y/%m/%d, %H:%M:%S', date)
    cache = dict(tag=tag, rev=rev, date=datestring)
    return cache

def getversion_svn(path=None):
    _program_dir = path or os.path.normpath(os.path.dirname(sys.argv[0]))
#   if not os.path.isabs(_program_dir):
#      _program_dir = os.path.normpath(os.path.join(os.getcwd(), _program_dir))
    entries = open(os.path.join(_program_dir, '.svn/entries'))
    version = entries.readline().strip()
    #use sqlite table for new entries format
    if version == "12":
        entries.close()
        from sqlite3 import dbapi2 as sqlite
        from datetime import datetime
        con = sqlite.connect(os.path.join(_program_dir, ".svn/wc.db"))
        cur = con.cursor()
        cur.execute( '''select local_relpath, repos_path, revision, changed_date from nodes order by revision desc, changed_date desc''')
        name, tag, rev, date = cur.fetchone()
        con.close()
        tag = tag[:-len(name)]
        date = time.gmtime(date/1000000)
    else:
        for i in xrange(3):
            entries.readline()
        tag = entries.readline().strip()
        t = tag.split('://')
        t[1] = t[1].replace('svn.wikimedia.org/svnroot/pywikipedia/', '')
        tag = '[%s] %s' % (t[0], t[1])
        for i in xrange(4):
            entries.readline()
        date = time.strptime(entries.readline()[:19],'%Y-%m-%dT%H:%M:%S')
        rev = entries.readline()[:-1]
        entries.close()
    if (not date or not tag or not rev) and not path:
        raise ParseError
    return (tag, rev, date)

def getversion_nightly():
    data = open(os.path.join(wikipediatools.get_base_dir(), 'version'))
    tag = data.readline().strip()
    date = time.strptime(data.readline()[:19],'%Y-%m-%dT%H:%M:%S')
    rev = data.readline().strip()
    if not date or not tag or not rev:
        raise ParseError
    return (tag, rev, date)

## Retrieve revision number of framework online repository's svnroot
#
def getversion_onlinerepo(repo=None):
    url = repo or 'http://svn.wikimedia.org/svnroot/pywikipedia/trunk/pywikipedia/'
    rev = None
    try:
        buf = urllib.urlopen(url).read()
        rev = buf.split(' ')[3][:-1]
    except:
        raise ParseError
    return rev

## Simple version comparison
#
cmp_ver = lambda a, b, tol=1: {-1: '<', 0: '~', 1: '>'}[cmp((a-b)//tol, 0)]

if __name__ == '__main__':
    print 'Pywikipedia %s' % getversion()
    print 'Python %s' % sys.version
    print 'config-settings:'
    print 'use_api =', config.use_api
    print 'use_api_login =', config.use_api_login
    if not __import__('unicodedata').normalize('NFC', u'\u092e\u093e\u0930\u094d\u0915 \u091c\u093c\u0941\u0915\u0947\u0930\u092c\u0930\u094d\u0917') == u'\u092e\u093e\u0930\u094d\u0915 \u091c\u093c\u0941\u0915\u0947\u0930\u092c\u0930\u094d\u0917':
        print u'unicode test: triggers problem #3081100'
    else:
        print u'unicode test: ok'

