
import os
import webapp2
import jinja2

from google.appengine.ext import db

template_dir = os.path.join(os.path.dirname(__file__), 'templates')
jinja_env = jinja2.Environment(loader = jinja2.FileSystemLoader(template_dir),
                               autoescape = True)

def render_str(template, **params):
	t = jinja_env.get_template(template)
	return t.render(params)

class Art(db.Model):
	title = db.StringProperty(required = True)
	art = db.TextProperty(required = True)
	created = db.DateTimeProperty(auto_now_add = True)
	

class Handler(webapp2.RequestHandler):
	def write(self, *a, **kw):
		self.response.out.write(*a,**kw)

	def render(self, template, **kw):
		self.write(render_str(template, **kw))

class MainPage(Handler):
	def render_front(self, title="", art="", error=""):
		arts = db.GqlQuery("SELECT * FROM Art ORDER BY created DESC")
		self.render("front.html", title=title, art=art, error=error, arts=arts)

	def get(self):
		self.render_front()

	def post(self):
		title = self.request.get("title")
		art = self.request.get("art")

		if title and art:
			a = Art(title = title, art = art)
			a.put()
			# key = a.key()
			# record = User.get(key)
			self.redirect("/")
			self.render_front()
		else:
			error = "Data not entered"
			self.render_front(title,art,error)
# 		items = self.request.get_all("food")
# 		self.render("shopping_lists.html" ,items = items)

# class FizzBuzzHandler(Handler):
# 	def get(self):
# 		self.render("shopping_lists.html")
# 		n= self.request.get('n',0)
# 		n=n and int(n)
# 		self.render('fizzbuzz.html',n=n)

app = webapp2.WSGIApplication([ ('/', MainPage)], debug=True)


