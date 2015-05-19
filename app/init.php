<?php
namespace app;
use library\utils, app\request, app\bootstrap;



class init{
    protected $error = false;
    protected $params;
    private $controller = __NAMESPACE__."\\controller\\";
 
    public function run(){
        utils::runMethods("app\bootstrap");
        $this->exceptionHandler();
        request::init();
        $this->dispatch();
    }
    public function dispatch(){
        $this->createController();
    }
    public function createController() {
        $this->params = array_merge(request::$params,request::$uri);
        $this->controller .= $this->params["controller"];
        if(request::hasError()){
          return new $this->controller($this->params["action"],$this->params,request::errorMsg());  
        }    
        return new $this->controller($this->params["action"],$this->params);

    }
    public function exceptionHandler(){
        set_exception_handler(array($this,"missedException"));
    }
    public static function missedException($e){
        echo "Sorry, something went wrong. Please try again or contact us if the problem persists.";
        utils::log("Unhandled exception: ".$e->getMessage().", file ".$e->getFile().", line ".$e->getLine());    
    }
    
}

 
