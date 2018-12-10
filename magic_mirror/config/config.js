/* Magic Mirror Config Sample
 *
 * By Michael Teeuw http://michaelteeuw.nl
 * MIT Licensed.
 *
 * For more information how you can configurate this file
 * See https://github.com/MichMich/MagicMirror#configuration
 *
 */

 var config = {
  address: "",          // Address to listen on, can be:
                        // - "localhost", "127.0.0.1", "::1" to listen on loopback interface
                        // - another specific IPv4/6 to listen on a specific interface
                        // - "", "0.0.0.0", "::" to listen on any interface
                        // Default, when address config is left out, is "localhost"
  port: 80,
  ipWhitelist: [],                                       // Set [] to allow all IP addresses
                                                         // or add a specific IPv4 of 192.168.1.5 :
                                                         // ["127.0.0.1", "::ffff:127.0.0.1", "::1", "::ffff:192.168.1.5"],
                                                         // or IPv4 range of 192.168.3.0 --> 192.168.3.15 use CIDR format :
                                                         // ["127.0.0.1", "::ffff:127.0.0.1", "::1", "::ffff:192.168.3.0/28"],

  language: "en",
  timeFormat: 24,
  units: "metric",

  modules: [
    {
      module: "alert",
    },
    {
      module: "updatenotification",
      position: "top_bar"
    },
    {
      module: "clock",
      position: "top_left"
    },
    {
      module: "calendar",
      header: "US Holidays",
      position: "top_left",
      config: {
        calendars: [
          {
            symbol: "calendar-check-o ",
            url: "webcal://www.calendarlabs.com/templates/ical/US-Holidays.ics"
          }
        ]
      }
    },
    {
      module: "compliments",
      position: "lower_third",
      config: {
        fadeSpeed: 0,
      }
    },
    {
      module: "currentweather",
      position: "top_right",
      config: {
        location: "Seattle",
        //locationID: "5809844",  //ID from http://bulk.openweathermap.org/sample/; unzip the gz file and find your city
        appid: "ff2b463393e18357bc80461de7a84296",
        animationSpeed: 0,
      }
    },
    {
      module: "weatherforecast",
      position: "top_right",
      header: "Weather Forecast",
      config: {
        location: "Seattle",
        //locationID: "5809844",
        appid: "ff2b463393e18357bc80461de7a84296",
        animationSpeed: 0,
      }
    },
    {
      module: "newsfeed",
      position: "bottom_bar",
      config: {
        feeds: [
          {
            title: "Hacker News",
            url: "https://news.ycombinator.com/rss"
          }
        ],
        showSourceTitle: true,
        showPublishDate: true,
        animationSpeed: 0,
      }
    },
  ]

};

/*************** DO NOT EDIT THE LINE BELOW ***************/
if (typeof module !== "undefined") {module.exports = config;}