# Manage team

The Team page allows you to manage who has access to the workspace. You can invite already existing
organization members to join your workspace or invite by email address where they will both join the
organization and workspace once they've accepted.

## Roles

There are two types of roles:

1. `Admin` - Able to do anything a member can and additionally invite new members, change roles or
   workspace settings
2. `Member` - Able to start compute, run queries and inspect results but not able to change settings
   or invite others.

There is also the notion of implicit and explicit roles. An organization admin is implicitly also a
workspace admin even though they are not an explicit member of the workspace. You can still add them
as an explicit member so they keep access to the workspace even when they are demoted down from
organization admin.

Workspace admins are able to promote others to admin or demote other admins to member. The system
enforces that there is always at least one workspace admin. In case you are unable to contact the
only admin of a workspace, you can ask an organization admin for help.
