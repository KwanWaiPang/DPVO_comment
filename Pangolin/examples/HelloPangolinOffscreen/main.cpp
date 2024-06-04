#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler.h>
#include <pangolin/gl/gldraw.h>

int main( int /*argc*/, char** /*argv*/ )
{
    static const int w = 640;
    static const int h = 480;

    pangolin::CreateWindowAndBind("Main",w,h,pangolin::Params({{"scheme", "headless"}}));
    glEnable(GL_DEPTH_TEST);

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(w,h,420,420,320,240,0.2,100),
        pangolin::ModelViewLookAt(-2,2,-2, 0,0,0, pangolin::AxisY)
    );

    // create a frame buffer object with colour and depth buffer
    pangolin::GlTexture color_buffer(w,h);
    pangolin::GlRenderBuffer depth_buffer(w,h);
    pangolin::GlFramebuffer fbo_buffer(color_buffer, depth_buffer);

    fbo_buffer.Bind();
    {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0,0,w,h);
        s_cam.Apply();

        // Render OpenGL Cube
        pangolin::glDrawColouredCube();

        // Swap frames and Process Events
        pangolin::FinishFrame();
    }
    fbo_buffer.Unbind();

    // download and save the colour buffer
    color_buffer.Save("fbo.png", false);

    return 0;
}
