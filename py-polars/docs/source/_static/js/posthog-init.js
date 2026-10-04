!(function (e, t) {
	var r, o, s, i;
	t.__SV ||
		((window.posthog = t),
		(t._i = []),
		(t.init = function (n, p, a) {
			function g(e, t) {
				var r = t.split(".");
				2 == r.length && ((e = e[r[0]]), (t = r[1])),
					(e[t] = function () {
						e.push([t].concat(Array.prototype.slice.call(arguments, 0)));
					});
			}
			((s = e.createElement("script")).type = "text/javascript"),
				(s.crossOrigin = "anonymous"),
				(s.async = !0),
				(s.src =
					p.api_host.replace(".i.posthog.com", "-assets.i.posthog.com") +
					"/static/array.js"),
				(i = e.getElementsByTagName("script")[0]).parentNode.insertBefore(s, i);
			var u = t;
			for (
				void 0 !== a ? (u = t[a] = []) : (a = "posthog"),
					u.people = u.people || [],
					u.toString = function (e) {
						var t = "posthog";
						return "posthog" !== a && (t += "." + a), e || (t += " (stub)"), t;
					},
					u.people.toString = function () {
						return u.toString(1) + ".people (stub)";
					},
					r =
						"init capture register register_once register_for_session unregister unregister_for_session getFeatureFlag getFeatureFlagResult isFeatureEnabled reloadFeatureFlags updateEarlyAccessFeatureEnrollment getEarlyAccessFeatures on onFeatureFlags onSessionId getSurveys getActiveMatchingSurveys renderSurvey canRenderSurvey getNextSurveyStep identify setPersonProperties group resetGroups setPersonPropertiesForFlags resetPersonPropertiesForFlags setGroupPropertiesForFlags resetGroupPropertiesForFlags reset get_distinct_id getGroups get_session_id get_session_replay_url alias set_config startSessionRecording stopSessionRecording sessionRecordingStarted captureException loadToolbar get_property getSessionProperty createPersonProfile opt_in_capturing opt_out_capturing has_opted_in_capturing has_opted_out_capturing clear_opt_in_out_capturing debug".split(
							" ",
						),
					o = 0;
				o < r.length;
				o++
			)
				g(u, r[o]);
			t._i.push([n, p, a]);
		}),
		(t.__SV = 1));
})(document, window.posthog || []);

var POSTHOG_ALLOWED_HOSTS = ["docs.pola.rs"];
if (POSTHOG_ALLOWED_HOSTS.includes(location.hostname)) {
	posthog.init("phc_uumWzEuHxsPf47UvXwi2N2TzeB3CbLURPTGGZNkyp6ZQ", {
		api_host: "https://iji.pola.rs",
		ui_host: "https://eu.posthog.com",
		defaults: "2026-05-30",
		cookieless_mode: "always",
		person_profiles: "never",
	});
}
