<?php
/**
 * Description of alert
 *
 * @author durso
 */
namespace library\layout\components;
use library\layout\components\component;
use library\layout\elements\link;
use library\layout\elements\inline;
use library\layout\elements\script;


class alert extends component{
    public function __construct($className = "alert-danger") {
        $this->attributes["class"] = array("alert",$className,"errorMsg");
        $this->tag = "div";
        $this->closeTag = true;
 
    }
    public function create($error){
        $a = new link("&times;");
        $a->addClassName("close");
        $a->setAttribute("data-dismiss","alert");
        $this->addChild($a);
        $span = new inline($error);
        $this->addChild($span);
        script::start();
    }
    
    
}