import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-8">
      {/* Header */}
      <header className="flex justify-between items-center mb-8">
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <div className="flex items-center gap-4">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Avatar className="cursor-pointer">
                <AvatarImage src="https://github.com/shadcn.png" />
                <AvatarFallback>CN</AvatarFallback>
              </Avatar>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              <DropdownMenuLabel>My Account</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem>Profile</DropdownMenuItem>
              <DropdownMenuItem>Settings</DropdownMenuItem>
              <DropdownMenuItem>Logout</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      {/* Main Content */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Welcome Card */}
        <Card>
          <CardHeader>
            <CardTitle>Welcome back!</CardTitle>
            <CardDescription>Here's what's happening with your projects</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <div className="w-full">
                  <Input placeholder="Search projects..." />
                </div>
                <Dialog>
                  <DialogTrigger asChild>
                    <Button>New Project</Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Create New Project</DialogTitle>
                      <DialogDescription>
                        Add a new project to your dashboard.
                      </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4 py-4">
                      <Input placeholder="Project name" />
                      <Input placeholder="Project description" />
                      <Button className="w-full">Create Project</Button>
                    </div>
                  </DialogContent>
                </Dialog>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Stats Card */}
        <Card>
          <CardHeader>
            <CardTitle>Statistics</CardTitle>
            <CardDescription>Your activity this month</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <p className="text-sm text-gray-500 dark:text-gray-400">Total Projects</p>
                <p className="text-3xl font-bold">12</p>
              </div>
              <div className="space-y-2">
                <p className="text-sm text-gray-500 dark:text-gray-400">Active Tasks</p>
                <p className="text-3xl font-bold">5</p>
              </div>
            </div>
          </CardContent>
          <CardFooter>
            <Button variant="outline" className="w-full">View Details</Button>
          </CardFooter>
        </Card>
      </div>
    </div>
  )
}
