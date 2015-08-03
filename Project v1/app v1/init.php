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
        $this->errorHandler();
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
    public function errorHandler(){
        register_shutdown_function(array($this,"missedError"));
    }
    public static function missedException($e){
        header('HTTP/1.1 500 Internal Server Error');
        echo "Sorry, something went wrong. Please try again or contact us if the problem persists.";
        echo "Unhandled exception: ".$e->getMessage().", file ".$e->getFile().", line ".$e->getLine();
        utils::log("Unhandled exception: ".$e->getMessage().", file ".$e->getFile().", line ".$e->getLine());    
    }
    public static function missedError(){
        $e = error_get_last();
        if($e['type'] === 1){ 
            header('HTTP/1.1 500 Internal Server Error');
            echo "Sorry, something went wrong. Please try again or contact us if the problem persists.";
        }
    }
    
}

 
