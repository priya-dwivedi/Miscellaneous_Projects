
import os
import webapp2
import jinja2
import hashlib
import re
from string import letters

import urllib2
from xml.dom import minidom

from google.appengine.ext import db

template_dir = os.path.join(os.path.dirname(__file__), 'templates')
jinja_env = jinja2.Environment(loader = jinja2.FileSystemLoader(template_dir),
                               autoescape = True)

# def hash_str(s):
# 	return hashlib.md5(s).hexdigest()

# def make_secure_val(s):
# 	return "%s|%s" % (s, hash_str(s))

# def check_secure_val(h):
# 	val = h.split('|')[0]
# 	hashed = h.split('|')[1]
# 	if hashed == hash_str(val):
# 		return val
# 	else:
# 		return 'None'

# IP_URl = "http://freegeoip.net/xml/" 
# def get_coords(ip):
# 	ip = "4.2.2.2"
# 	url = IP_URl + ip
# 	content = None
# 	try:
# 		content = urllib2.urlopen(url).read()
# 	except:
# 		return 

# 	if content:
# 		x = minidom.parseString(content)
# 		lat = x.getElementsByTagName("Latitude")[0].childNodes[0].nodeValue
# 		lon = x.getElementsByTagName("Longitude")[0].childNodes[0].nodeValue
# 		return db.GeoPt(lat,lon)
# 	else:
# 		return "None"

temp_user = ''

# Define base handler class
class Handler(webapp2.RequestHandler):
    def write(self, *a, **kw):
        self.response.out.write(*a, **kw)
    def render_str(self, template, **params):
        t = jinja_env.get_template(template)
        return t.render(params)
    def render(self, template, **kw):
        self.write(self.render_str(template, **kw))

# Code for sign up
## Email address and Password entered must be part of database

allowed = {'priya': 'sneha', 'marie.mizgala@scotiabank.com': 's6324800'}

class Signup(Handler):

    def get(self):
        self.render("signup-form3.html")

    def post(self):
        have_error = False
        email = self.request.get('email')
        password = self.request.get('password')

        params = dict(email = email,
                      password = password)

        if allowed.has_key(email):
			if allowed[email] == password:
				have_error = False
			else:
				have_error = True
				params['error_password'] = "That's not a valid password."
        else:
        	have_error = True
        	params['error_email'] = "That's not a valid email."

        if have_error:
            self.render('signup-form3.html', **params)
        else:
        	username = email.split('.')[0]
        	self.write(username)
        	temp_user = username

        	self.redirect('/mainpage')


class Case(db.Model):
    application = db.StringProperty(required = True)
    insurer = db.StringProperty(required = True)
    date = db.StringProperty(required = True)
    ins_comment = db.TextProperty(required=True)
    inv_comment = db.TextProperty(required = False)
    created = db.DateTimeProperty(auto_now_add = True)

def top_case():
	cases = db.GqlQuery("SELECT * FROM Case ORDER BY created DESC LIMIT 50 ")
	cases = list(cases)
	return (cases)


class MainHandler(Handler):
    def render_front(self, application="", insurer = "", date = "", ins_comment = "", inv_comment="", username = "", error=""):
        cases = db.GqlQuery("SELECT * FROM Case ORDER BY created DESC LIMIT 50 ")
        username = temp_user
        self.render("edd_display2.html", application=application, insurer=insurer, date = date, ins_comment = ins_comment, inv_comment = inv_comment,
        	username = str(username), error=error, cases = cases)

    def get(self):
		return self.render_front()

    def post(self):
        application = self.request.get("application")
        insurer = self.request.get("insurer")
        date = self.request.get("date")
        ins_comment = self.request.get("ins_comment")
        inv_comment = self.request.get("inv_comment")

        if not inv_comment:
        	inv_comment = ""

        if application and insurer and date and ins_comment:
            a = Case(application = application, insurer = insurer, date = date, ins_comment = ins_comment, inv_comment=inv_comment)
            a.put()
            self.redirect("/mainpage")
            self.render_front()
        else:
            error = "Need first four fields"
            self.render_front(application, insurer, date, ins_comment, inv_comment, error) 

### Add the new handler
class ModHandler(Handler):
    def render_front(self, application="", insurer = "", date = "", ins_comment = "", inv_comment="", username = "", error=""):
        cases = db.GqlQuery("SELECT * FROM Case ORDER BY created DESC LIMIT 50 ")
        username = temp_user
        self.render("edd_modify2.html", application=application, insurer=insurer, date = date, ins_comment = ins_comment, inv_comment = inv_comment,
            username = str(username), error=error, cases = cases)

    def get(self):
        return self.render_front()

    def post(self):
        if self.request.get("ex_application"):
            new_application = self.request.get("ex_application")
            new_insurer = self.request.get("ex_insurer")
            new_date = self.request.get("ex_date")
            new_ins_comment = self.request.get("ex_ins_comment")
            new_inv_comment = self.request.get("ex_inv_comment")
            for case in cases:
                if case.application == new_application:
                    if ((case.insurer != new_insurer) or (case.date != new_date) or(case.ins_comment!= new_ins_comment) or (case.inv_comment != new_inv_comment)):
                        a = Case(application = case.application, insurer = case.insurer, date = case.date, ins_comment = case.ins_comment, inv_comment= case.inv_comment)
                        b = Case(application = new_application, insurer = new_insurer, date = new_date, ins_comment = new_ins_comment, inv_comment= new_inv_comment)
                        b.put()
                        a.delete()
        self.redirect("/mainpage")
        # self.redirect("/modcases")
        # self.render_front()


app = webapp2.WSGIApplication([ ('/signup', Signup),
								('/mainpage', MainHandler),
                                ('/modcases', ModHandler)], 
								debug=True)


