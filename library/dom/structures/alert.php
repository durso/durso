<?php
/**
 * Description of alert
 *
 * @author durso
 */
namespace library\dom\structures;
use library\dom\structures\components;
use library\dom\elements\components\link;
use library\dom\elements\components\inline;
use library\dom\elements\components\block;



class alert extends components{
    private $a;
    private $span;
    
    public function __construct($className = "alert-danger") {
        $this->root = new block("div");
        $this->root->addClass("alert $className errorMsg");
    }
    public function create($error){
        $this->a = new link("&times;");
        $this->a->addClass("close");
        $this->a->attr("data-dismiss","alert");
        $this->root->addComponent($this->a);
        $this->components["a"][] = $this->a;
        $this->span = new inline("span",$error);
        $this->root->addComponent($this->span);
        $this->components["span"][] = $this->span;
    }
    public function save(){
        return $this->root;
    }
    
    
}